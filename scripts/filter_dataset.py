#!/usr/bin/env python3
# 2-space indentation, strict typing
import argparse
import gzip
import io
import sys
import json
import heapq
import hashlib
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import os
from concurrent.futures import ProcessPoolExecutor

ROOT: Path = Path(__file__).resolve().parents[1]
src_path = str(ROOT / "src")
if src_path not in sys.path:
  sys.path.insert(0, src_path)

from core.board import Board
from core.enums import GameState

"""
Filter a selfplay JSONL(.gz) dataset to keep the top-N higher-quality games.

Heuristic (tunable):
- Prefer NN > NR > R groups
- Prefer terminal results (W/B > D > null)
- Prefer longer games (up to a cap)
- Drop exact duplicates by move transcript (optional)
- Optionally enforce Base+MLP only
- Optional 70/20/10 mix preservation with shortfall fill from the global best

Output format matches scripts/selfplay_to_jsonl.py.

Usage:
  python scripts/filter_dataset_by_quality.py \
    --in data/selfplay.jsonl.gz --out data/selfplay.top.jsonl.gz --target 200000 \
    --variant Base+MLP --preserve-mix --min-plies 8
"""

# ------------------------------ CLI -----------------------------------
@dataclass
class Args(argparse.Namespace):
  inp: str
  out: str
  target: int
  variant: Optional[str]
  preserve_mix: bool
  min_plies: int
  dedup: bool
  seed: int
  max_plies_ref: int
  workers: int
  chunksize: int


def parse_args(argv: Optional[Iterable[str]] = None) -> Args:
  p = argparse.ArgumentParser(description="Filter selfplay JSONL(.gz) to top-N by a quality heuristic")
  p.add_argument("--in", dest="inp", required=True, help="input .jsonl(.gz)")
  p.add_argument("--out", required=True, help="output .jsonl(.gz)")
  p.add_argument("--target", type=int, required=True, help="number of games to keep")
  p.add_argument("--variant", default=None, help="keep only games whose header equals/contains this token (e.g., Base+MLP)")
  p.add_argument("--preserve-mix", action="store_true", help="preserve 70/20/10 R/NR/NN counts (shortfalls auto-filled from global best)")
  p.add_argument("--min-plies", type=int, default=0, help="drop games shorter than this many plies")
  p.add_argument("--dedup", action="store_true", default=True, help="drop exact duplicate move transcripts (default on)")
  p.add_argument("--no-dedup", dest="no_dedup", action="store_true", help="disable dedup of move transcripts")
  p.add_argument("--seed", type=int, default=0, help="tie-breaker seed for stable ordering")
  p.add_argument("--max-plies-ref", type=int, default=300, help="reference max plies for length scoring")
  p.add_argument("--workers", type=int, default=(os.cpu_count() or 1), help="number of parallel worker processes")
  p.add_argument("--chunksize", type=int, default=64, help="batch size per worker for map()")
  a = p.parse_args(list(argv) if argv is not None else None)
  # normalize dedup flags
  dedup = True
  if a.no_dedup:
    dedup = False
  elif a.dedup:
    dedup = True
  return Args(a.inp, a.out, a.target, a.variant, a.preserve_mix, a.min_plies, dedup, a.seed, a.max_plies_ref, workers=a.workers, chunksize=a.chunksize)


# ----------------------------- I/O utils -------------------------------

def _is_gz(path: Path) -> bool:
  return path.suffix == ".gz" or path.name.endswith(".jsonl.gz")


def _open_reader(p: Path) -> io.TextIOBase:
  if _is_gz(p):
    return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8")
  return open(p, "r", encoding="utf-8")


def _open_writer(p: Path) -> io.TextIOBase:
  p.parent.mkdir(parents=True, exist_ok=True)
  if _is_gz(p):
    return io.TextIOWrapper(gzip.open(p, "wb"), encoding="utf-8")
  return open(p, "w", encoding="utf-8")


# --------------------------- Scoring -----------------------------------
GroupWeights: Dict[str, float] = {"R": 1.0, "NR": 2.0, "NN": 3.0}


def _result_bonus(res: Optional[str]) -> float:
  if res == "W" or res == "B":
    return 0.5
  if res == "D":
    return 0.25
  return 0.0


def _len_bonus(ply_count: int, max_ref: int) -> float:
  # up to +0.5 as games approach max_ref plies
  if ply_count <= 0:
    return 0.0
  x = min(ply_count / max(1, max_ref), 1.0)
  return 0.5 * x


def _short_penalty(ply_count: int) -> float:
  return -0.25 if ply_count < 20 else 0.0


def _map_result(state: GameState) -> Optional[str]:
  if state is GameState.DRAW:
    return "D"
  if state is GameState.WHITE_WINS:
    return "W"
  if state is GameState.BLACK_WINS:
    return "B"
  return None

def _sanity_ok(rec: Dict[str, Any]) -> bool:
  # basic schema checks
  mv = rec.get("moves")
  if not isinstance(mv, list):
    return False
  if rec.get("ply_count") != len(mv):
    return False
  # replay on a fresh Board
  try:
    b = Board("Base+MLP")
    for m in mv:
      try:
        b.play(m)
      except Exception:
        return False
    # canonical gamestring must match
    if rec.get("game_string") != str(b):
      return False
    # if terminal, result must match; otherwise accept any (typically None)
    exp = _map_result(b.state)
    if exp is not None and rec.get("result") != exp:
      return False
    return True
  except Exception:
    return False


# -------------------------- Filtering core -----------------------------
class Reservoir:
  def __init__(self, cap: int) -> None:
    self.cap = cap
    self.heap: List[Tuple[float, int, str]] = []  # (score, tiebreaker, json_line)
    self.count_seen = 0

  def push(self, score: float, tiebreaker: int, line: str) -> None:
    if self.cap <= 0:
      return
    item = (score, tiebreaker, line)
    if len(self.heap) < self.cap:
      heapq.heappush(self.heap, item)
    else:
      if item > self.heap[0]:
        heapq.heapreplace(self.heap, item)
    self.count_seen += 1

  def items_desc(self) -> List[str]:
    return [x[2] for x in sorted(self.heap, key=lambda t: (t[0], t[1]), reverse=True)]


def _split_counts(total: int) -> Tuple[int, int, int]:
  r = int(total * 0.70)
  nr = int(total * 0.20)
  nn = total - r - nr
  return r, nr, nn


def _variant_ok(gs: str, token: Optional[str]) -> bool:
  if not token:
    return True
  head = gs.split(";", 1)[0].strip()
  return token == head or token in head


def _process_line_job(args: Tuple[int, str, Optional[str], int, int]) -> Optional[Tuple[int, str, str, str, float]]:
  seq, line, variant_token, min_plies, max_ref = args
  line_stripped = line.strip()
  if not line_stripped:
    return None
  try:
    rec = json.loads(line_stripped)
  except Exception:
    return None
  gs = rec.get("game_string")
  if not isinstance(gs, str) or not _variant_ok(gs, variant_token):
    return None
  ply = int(rec.get("ply_count", 0) or 0)
  if ply < min_plies:
    return None
  mv = rec.get("moves")
  if not isinstance(mv, list):
    return None

  # exact-moves transcript key (for global dedup in the reducer)
  key = hashlib.sha1((" ".join(map(str, mv))).encode("utf-8")).hexdigest()

  # sanity check
  if not _sanity_ok(rec):
    return None

  meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
  group = meta.get("group") if isinstance(meta, dict) else None
  if group not in ("R", "NR", "NN"):
    group = "R"
  base = GroupWeights.get(group, 1.0)
  res = rec.get("result") if isinstance(rec.get("result"), str) else None
  score = base + _result_bonus(res) + _len_bonus(ply, max_ref) + _short_penalty(ply)

  return (seq, (line_stripped if line_stripped.endswith("\n") else line_stripped + "\n"), key, group, score)


def filter_dataset(a: Args) -> int:
  inp = Path(a.inp).resolve()
  outp = Path(a.out).resolve()
  if not inp.exists():
    print(f"missing input: {inp}")
    return 2
  outp.parent.mkdir(parents=True, exist_ok=True)

  # capacities
  if a.preserve_mix:
    r_cap, nr_cap, nn_cap = _split_counts(a.target)
  else:
    r_cap = nr_cap = nn_cap = 0
  global_cap = a.target

  # reservoirs
  R = Reservoir(r_cap)
  NR = Reservoir(nr_cap)
  NN = Reservoir(nn_cap)
  G = Reservoir(global_cap)

  # dedup set
  seen: set[str] = set()

  total_in = 0
  kept_candidates = 0

  def _jobs(reader: io.TextIOBase):
    for seq, line in enumerate(reader, start=1):
      if not line.strip():
        continue
      yield (seq, line, a.variant, a.min_plies, a.max_plies_ref)

  with _open_reader(inp) as rd, ProcessPoolExecutor(max_workers=a.workers) as ex:
    for out in ex.map(_process_line_job, _jobs(rd), chunksize=a.chunksize):
      if out is None:
        continue
      total_in += 1
      seq, line_norm, key, group, score = out
      if a.dedup:
        if key in seen:
          continue
        seen.add(key)
      tiebreak = seq  # stable tie-breaker by input order
      if a.preserve_mix:
        if group == "R":
          R.push(score, tiebreak, line_norm)
        elif group == "NR":
          NR.push(score, tiebreak, line_norm)
        else:
          NN.push(score, tiebreak, line_norm)
      G.push(score, tiebreak, line_norm)
      kept_candidates += 1

  # assemble selection
  selected: List[str] = []
  if a.preserve_mix:
    for pool in (R, NR, NN):
      selected.extend(pool.items_desc())
  # fill shortfall from global
  if len(selected) < a.target:
    # build a set of ids of already selected lines to avoid dup; compare by full line string hash
    selected_set = set(hash(s) for s in selected)
    for s in G.items_desc():
      if len(selected) >= a.target:
        break
      if hash(s) in selected_set:
        continue
      selected.append(s)
      selected_set.add(hash(s))
  # trim if somehow exceeded
  if len(selected) > a.target:
    selected = selected[: a.target]

  # write out
  with _open_writer(outp) as out:
    for s in selected:
      out.write(s if s.endswith("\n") else s + "\n")

  print(f"input: {total_in} lines  → candidates: {kept_candidates}  → written: {len(selected)} → {outp}")
  return 0 if len(selected) > 0 else 2


# ------------------------------ Main -----------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
  a = parse_args(argv)
  return filter_dataset(a)


if __name__ == "__main__":
  raise SystemExit(main())
