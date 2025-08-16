#!/usr/bin/env python3
import argparse
import dataclasses as dc
import gzip
import io
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Callable, Protocol

"""
Self-play generator → JSONL of UHP GameStrings using your Hivemind engine.

Strict typing. 2-space indentation.

Mix is fixed:
  70% random vs random
  20% negamax vs random (sides alternate)
  10% negamax vs negamax

Each JSON line has:
{
  "game_string": str,
  "moves": [str,...],
  "result": "W"|"B"|"D"|null,
  "ply_count": int,
  "meta": {"start_ts":..., "seed":..., "engine": ..., "group": "R|NR|NN"}
}

Usage:
  python scripts/selfplay_to_jsonl.py --games 10000 --out data/selfplay.jsonl.gz --workers 20 --negamax-movetime 5

Validation only:
  python scripts/selfplay_to_jsonl.py --validate-path data/selfplay.jsonl.gz --sample 100
"""

# ----------------- Import engine from ./src -----------------
ROOT: Path = Path(__file__).resolve().parents[1]
src_path = str(ROOT / "src")
if src_path not in sys.path:
  sys.path.insert(0, src_path)

from core.board import Board
from core.enums import PlayerColor, GameState
from engine import Engine
from ai.brain import AlphaBetaPruner
from copy import deepcopy


# ------------------------------ Agents ---------------------------------
class AgentProtocol(Protocol):
  def select(self, adapter: "EngineAdapter", color: PlayerColor, legal_moves: list[str]) -> str: ...


class RandomAgent:
  def select(self, adapter: "EngineAdapter", color: PlayerColor, legal_moves: list[str]) -> str:
    return random.choice(legal_moves)


class BrainNegamaxAgent:
  """Alpha-beta agent using ai.brain.AlphaBetaPruner with a time limit."""
  def __init__(self, movetime_s: float = 5.0, max_branching: int | None = None, max_depth: int = 0) -> None:
    self.movetime_s: int = int(movetime_s)           # seconds per move
    self.max_depth: int = int(max_depth)             # 0 = iterative deepening until time
    self.max_branching: int = (
      max_branching if max_branching is not None else Engine.DEFAULT_MAX_BRANCHING_FACTOR
    )
    self._brain: AlphaBetaPruner = AlphaBetaPruner()

  def select(self, adapter: "EngineAdapter", color: PlayerColor, legal_moves: list[str]) -> str:
    board_copy = deepcopy(adapter.board())  # search mutates
    move: str = self._brain.find_best_move(
      board_copy,
      self.max_branching,
      time_limit=self.movetime_s,
    )
    return move if move in legal_moves else random.choice(legal_moves)


# --------------------------- Validation utils ---------------------------
class _Validator:
  def __init__(self) -> None:
    pass

  @staticmethod
  def _is_terminal(board: Board) -> tuple[bool, Optional[str]]:
    if board.state is GameState.DRAW:
      return True, "D"
    if board.state is GameState.WHITE_WINS:
      return True, "W"
    if board.state is GameState.BLACK_WINS:
      return True, "B"
    return False, None

  @staticmethod
  def check_record(rec: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(rec.get("moves"), list):
      return False, "moves not a list"
    if rec.get("ply_count") != len(rec["moves"]):
      return False, "ply_count mismatch"

    b = Board()
    for i, mv in enumerate(rec["moves"]):
      legal = b.valid_moves.split(";")
      if mv not in legal:
        return False, f"illegal move at ply {i}: {mv}"
      b.play(mv)

    gs = str(b)
    if rec.get("game_string") != gs:
      return False, "gamestring mismatch after replay"

    term, res = _Validator._is_terminal(b)
    if term:
      if rec.get("result") != res:
        return False, f"result mismatch: got {rec.get('result')}, expected {res}"
    return True, "ok"


def validate_jsonl(path: str, sample: int = 0) -> int:
  """Return number of invalid records. sample=0 → validate all."""
  open_fn: Callable[[str], io.TextIOBase]
  if path.endswith(".gz"):
    def _open(p: str) -> io.TextIOBase:
      return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8")
    open_fn = _open
  else:
    open_fn = lambda p: open(p, "r", encoding="utf-8")  # type: ignore[assignment]

  rng = random.Random(0)
  n_bad = 0
  n = 0
  V = _Validator()
  with open_fn(path) as fh:
    for line in fh:
      if not line.strip():
        continue
      n += 1
      if sample and rng.randrange(sample) != 0:
        continue
      try:
        rec = json.loads(line)
      except Exception:
        n_bad += 1
        print(f"line {n}: invalid json")
        continue
      ok, msg = V.check_record(rec)
      if not ok:
        n_bad += 1
        print(f"line {n}: {msg}")
  print(f"validated {n} lines, invalid: {n_bad}")
  return n_bad


# ---------------------- Hivemind engine adapter ------------------------
class EngineAdapter:
  """Thin wrapper over your Board API with UHP I/O.

  Board(gamestring: str = "")
    .calculate_valid_moves() -> set[Move]
    .stringify_move(move: Move | None) -> str
    .play(move_string: str) -> None
    .queen_neighbors_by_color(color: PlayerColor) -> int
  """

  def __init__(self) -> None:
    self._board: Board | None = None
    self._moves_so_far: list[str] = []

  def new_game(self) -> None:
    self._board = Board("Base+MLP")
    self._moves_so_far = []

  def legal_moves(self) -> list[str]:
    assert self._board is not None
    move_objs = self._board.calculate_valid_moves()
    if not move_objs:
      return ["pass"]
    return sorted(self._board.stringify_move(m) for m in move_objs)

  def play(self, move: str) -> None:
    assert self._board is not None
    self._board.play(move)
    self._moves_so_far.append(move)

  def is_terminal(self) -> bool:
    assert self._board is not None
    return self._board.gameover

  def result(self) -> Optional[str]:
    assert self._board is not None
    if self._board.state is GameState.DRAW:
      return "D"
    if self._board.state is GameState.WHITE_WINS:
      return "W"
    if self._board.state is GameState.BLACK_WINS:
      return "B"
    return None

  def game_string(self) -> str:
    assert self._board is not None
    return str(self._board)

  def board(self) -> Board:
    assert self._board is not None
    return self._board


# ---------------------------- Self-play ---------------------------------
@dc.dataclass(slots=True)
class GameRecord:
  game_string: str
  moves: list[str]
  result: Optional[str]
  ply_count: int
  meta: dict[str, Any]


def play_one(
  adapter: EngineAdapter,
  max_plies: int,
  agent_w: AgentProtocol,
  agent_b: AgentProtocol,
  seed: int,
  on_step: Callable[[int], None] | None = None,
) -> GameRecord:
  random.seed(seed)
  adapter.new_game()
  moves: list[str] = []

  for ply in range(max_plies):
    if adapter.is_terminal():
      break
    legal = adapter.legal_moves()
    color = PlayerColor.WHITE if ply % 2 == 0 else PlayerColor.BLACK
    move = (agent_w if ply % 2 == 0 else agent_b).select(adapter, color, legal)
    adapter.play(move)
    moves.append(move)
    if on_step:
      on_step(ply + 1)  # 1-based plies completed

  return GameRecord(
    game_string=adapter.game_string(),
    moves=moves,
    result=adapter.result(),
    ply_count=len(moves),
    meta={},
  )


# ----------------------------- I/O utils -------------------------------
class JsonlWriter:
  def __init__(self, path: str):
    self.path = path
    self._fh: io.TextIOBase
    if path.endswith(".gz"):
      self._fh = io.TextIOWrapper(gzip.open(path, "wb"), encoding="utf-8")
    else:
      self._fh = open(path, "w", encoding="utf-8")

  def write(self, obj: Any) -> None:
    self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

  def close(self) -> None:
    self._fh.close()


# ------------------------------- CLI -----------------------------------
@dc.dataclass(slots=True)
class Args:
  games: int
  out: str
  max_plies: int
  seed: int
  workers: int
  validate_path: Optional[str]
  sample: int
  negamax_movetime: float


def parse_args(argv: Optional[Iterable[str]] = None) -> Args:
  p = argparse.ArgumentParser(description="Generate or validate self-play UHP JSONL with a fixed game mix: 70% random, 20% negamax vs random, 10% negamax vs negamax")
  p.add_argument("--games", type=int, default=0, help="total number of games to generate; 0 → skip generation")
  p.add_argument("--out", type=str, default="data/selfplay.jsonl.gz", help="output .jsonl or .jsonl.gz path for generation")
  p.add_argument("--max-plies", type=int, default=240, help="ply cap per game")
  p.add_argument("--seed", type=int, default=42, help="RNG seed")
  p.add_argument("--workers", type=int, default=1, help="processes for parallel generation")
  p.add_argument("--validate-path", dest="validate_path", type=str, default=None, help="validate an existing JSONL(.gz) and exit")
  p.add_argument("--sample", type=int, default=0, help="validate 1/sample of lines (0 = all)")
  p.add_argument("--negamax-movetime", dest="negamax_movetime", type=float, default=5.0, help="seconds per move for negamax agents")
  a = p.parse_args(list(argv) if argv is not None else None)
  return Args(a.games, a.out, a.max_plies, a.seed, a.workers, a.validate_path, a.sample, a.negamax_movetime)


# ----------------------- Parallel generation ---------------------------

def _fmt_eta(seconds: float) -> str:
  if seconds <= 0 or seconds == float("inf"):
    return "ETA —"
  total = int(seconds + 0.5)
  m, s = divmod(total, 60)
  h, m = divmod(m, 60)
  return f"ETA {h:02d}:{m:02d}:{s:02d}"


def _read_progress(path: str) -> dict[str, Any] | None:
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)
  except Exception:
    return None


def _initial_sec_per_game(group: str, movetime: float, max_plies: int) -> float:
  k = 0.0 if group == "R" else (0.5 if group == "NR" else 1.0)
  # conservative upper bound for NR/NN dominates early ETA; R stays tiny
  cons = max_plies * max(0.001, k * movetime)
  prior = cons * 1.10  # small overhead
  return max(prior, 0.02 * max_plies)  # floor to avoid zero for R


def _sec_per_game_est(group: str, avg_sec: float | None, movetime: float, max_plies: int, prior_sec: float) -> float:
  k = 0.0 if group == "R" else (0.5 if group == "NR" else 1.0)
  conservative = max_plies * max(0.001, k * movetime)
  if group in ("NR", "NN"):
    if avg_sec is None:
      # heavily favor conservative bound over prior early on
      return 0.9 * conservative + 0.1 * prior_sec
    # blend toward measured EWMA but keep some conservatism
    return 0.7 * avg_sec + 0.3 * conservative
  # Random group: use EWMA if available, else prior
  return avg_sec if avg_sec is not None else prior_sec


def _eta_group(done_float: float, total: int, workers: int, avg_sec: float | None, prior_sec: float, group: str, movetime: float, max_plies: int) -> float:
  remaining_games = max(0.0, total - done_float)
  sec_per_game = _sec_per_game_est(group, avg_sec, movetime, max_plies, prior_sec)
  return (remaining_games * sec_per_game) / max(1, workers)


def _progress_line2(tag: str, done_float: float, total: int, eta_sec: float) -> str:
  pct = (done_float / total * 100.0) if total else 100.0
  return f"[{tag:2}] {int(done_float)}/{total} ({pct:5.1f}%)  {_fmt_eta(eta_sec)}"


def _worker_run_group(worker_id: int, group: str, n_games: int, max_plies: int, seed: int, part_path: str, meta: dict[str, Any], negamax_movetime: float) -> dict[str, Any]:
  rnd = random.Random(seed ^ (worker_id * 0x9E3779B1))
  adapter = EngineAdapter()
  writer = JsonlWriter(part_path)
  t0 = time.time()
  nm = BrainNegamaxAgent(negamax_movetime)
  rnd_agent = RandomAgent()
  prog_path = part_path + ".prog"
  err_part_path = part_path + ".errors"

  # priors: seconds per game (conservative already baked in)
  prior_sec_per_game = _initial_sec_per_game(group, negamax_movetime, max_plies)

  avg_sec: float | None = None
  games_done = 0

  def write_prog(done_float: float, cur_ply: int, est_plies: int, avg: float | None) -> None:
    obj: dict[str, float | int | str | None] = {
      "done_float": done_float,
      "done": int(done_float),
      "total": n_games,
      "t0": t0,
      "group": group,
      "worker": worker_id,
      "cur_ply": cur_ply,
      "est_plies": est_plies,
      "avg_sec_per_game": avg,
      "prior_sec_per_game": prior_sec_per_game,
    }
    # Windows-safe: write directly, flush, and ignore sharing violations.
    try:
      with open(prog_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    except PermissionError:
      pass
    except OSError:
      pass

  est_plies = max_plies
  # generate until we have exactly n_games successful records
  while games_done < n_games:
    idx = games_done  # stable index for parity and meta
    game_seed = rnd.randrange(2**31)
    random.seed(game_seed)
    if group == "R":
      aw: AgentProtocol = rnd_agent; ab: AgentProtocol = rnd_agent
    elif group == "NR":
      aw, ab = (nm, rnd_agent) if idx % 2 == 0 else (rnd_agent, nm)
    else:
      aw, ab = nm, nm

    t_start = time.time()
    try:
      rec = play_one(
        adapter, max_plies, aw, ab, seed=game_seed,
        on_step=lambda p: write_prog(games_done + p / float(est_plies), p, est_plies, avg_sec),
      )
    except Exception as e:
      # Log error and restart a new game "in its place"
      try:
        gs = adapter.game_string()
      except Exception:
        gs = "<unavailable>"
      try:
        with open(err_part_path, "a", encoding="utf-8") as ef:
          ef.write(gs + "\n" + repr(e) + "\n\n")
      except OSError:
        pass
      # reset fractional progress for this slot
      write_prog(float(games_done), 0, est_plies, avg_sec)
      continue  # do not advance games_done; try again

    dt = time.time() - t_start
    avg_sec = dt if avg_sec is None else 0.7 * avg_sec + 0.3 * dt

    rec.meta = {**meta, "worker": worker_id, "group": group, "idx": idx, "game_seed": game_seed, "secs": dt}
    writer.write(dc.asdict(rec))

    games_done += 1
    write_prog(float(games_done), 0, est_plies, avg_sec)

  writer.close()
  write_prog(float(n_games), 0, est_plies, avg_sec)
  return {"worker": worker_id, "group": group, "games": n_games, "secs": time.time() - t0}


def _concat_parts(parts: list[str], final_path: str) -> None:
  if final_path.endswith(".gz"):
    with open(final_path, "wb") as outb:
      for pp in parts:
        with open(pp, "rb") as f:
          outb.write(f.read())
  else:
    with open(final_path, "w", encoding="utf-8") as outt:
      for pp in parts:
        opener = gzip.open if pp.endswith(".gz") else open
        mode = "rt" if pp.endswith(".gz") else "r"
        with opener(pp, mode) as f:
          for line in f:
            outt.write(line) # type: ignore


def _error_final_path(out_path: str) -> str:
  d = os.path.dirname(out_path) or "."
  name = os.path.basename(out_path)
  for ext in (".gz", ".jsonl", ".json"):
    if name.endswith(ext):
      name = name[: -len(ext)]
  return os.path.join(d, f"{name}-errors.txt")


def _merge_error_parts(part_paths: list[str], final_out: str) -> int:
  """Merge per-part .errors files into data/<filename>-errors.txt.
  Returns approximate number of cases merged.
  """
  err_out = _error_final_path(final_out)
  os.makedirs(os.path.dirname(err_out) or ".", exist_ok=True)
  total_cases = 0
  with open(err_out, "w", encoding="utf-8") as out:
    for pp in part_paths:
      ep = pp + ".errors"
      if not os.path.exists(ep):
        continue
      try:
        with open(ep, "r", encoding="utf-8") as f:
          data = f.read()
        if data:
          out.write(data)
          # crude estimate: blank line separates cases
          total_cases += data.count("\n\n")
      except OSError:
        pass
  if total_cases == 0:
    try:
      os.remove(err_out)
    except OSError:
      pass
  return total_cases


def _split_counts(total: int) -> tuple[int, int, int]:
  r = int(total * 0.99)
  nr = int(total * 0.0075)
  nn = total - r - nr
  return r, nr, nn


def generate_parallel(args: Args) -> None:
  assert args.out, "--out is required for generation"
  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  try:
    engine_version = getattr(Engine, "VERSION", "unknown")
  except Exception:
    engine_version = "unknown"
  meta: dict[str, str | int | float] = {
    "start_ts": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
    "seed": args.seed,
    "engine": f"Hivemind {engine_version}",
    "negamax_movetime": args.negamax_movetime,
  }

  r_count, nr_count, nn_count = _split_counts(args.games)

  # Single-process path with live progress (per-game granularity)
  if args.workers <= 1:
    writer = JsonlWriter(args.out)
    rnd = random.Random(args.seed)
    adapter = EngineAdapter()
    nm = BrainNegamaxAgent(args.negamax_movetime)
    rnd_agent = RandomAgent()

    # R group
    last = 0.0
    for g in range(r_count):
      seed_g = rnd.randrange(2**31)
      rec = play_one(adapter, args.max_plies, rnd_agent, rnd_agent, seed=seed_g)
      rec.meta = {**meta, "worker": 0, "group": "R", "idx": g, "game_seed": seed_g}
      writer.write(dc.asdict(rec))
      now = time.time()
      if now - last >= 10.0:
        sys.stdout.write("\n" + f"[R ] {g + 1}/{r_count} ({(g + 1) / max(1, r_count) * 100:5.1f}%)")
        sys.stdout.flush()
        last = now
    if r_count:
      sys.stdout.write("\n" + f"[R ] {r_count}/{r_count} (100.0%)\n"); sys.stdout.flush()

    # NR group
    last = 0.0
    for g in range(nr_count):
      seed_g = rnd.randrange(2**31)
      aw, ab = (nm, rnd_agent) if g % 2 == 0 else (rnd_agent, nm)
      rec = play_one(adapter, args.max_plies, aw, ab, seed=seed_g)
      rec.meta = {**meta, "worker": 0, "group": "NR", "idx": g, "game_seed": seed_g}
      writer.write(dc.asdict(rec))
      now = time.time()
      if now - last >= 10.0:
        sys.stdout.write("\n" + f"[NR] {g + 1}/{nr_count} ({(g + 1) / max(1, nr_count) * 100:5.1f}%)")
        sys.stdout.flush()
        last = now
    if nr_count:
      sys.stdout.write("\n" + f"[NR] {nr_count}/{nr_count} (100.0%)\n"); sys.stdout.flush()

    # NN group
    last = 0.0
    for g in range(nn_count):
      seed_g = rnd.randrange(2**31)
      rec = play_one(adapter, args.max_plies, nm, nm, seed=seed_g)
      rec.meta = {**meta, "worker": 0, "group": "NN", "idx": g, "game_seed": seed_g}
      writer.write(dc.asdict(rec))
      now = time.time()
      if now - last >= 10.0:
        sys.stdout.write("\n" + f"[NN] {g + 1}/{nn_count} ({(g + 1) / max(1, nn_count) * 100:5.1f}%)")
        sys.stdout.flush()
        last = now
    if nn_count:
      sys.stdout.write("\n" + f"[NN] {nn_count}/{nn_count} (100.0%)\n"); sys.stdout.flush()

    writer.close()
    print(f"done → {args.out}")
    return

  # Multi-process path with aggregated live progress (fractional)
  parts_dir = args.out + ".parts"
  os.makedirs(parts_dir, exist_ok=True)
  # remove stale temp files from previous runs
  try:
    for name in os.listdir(parts_dir):
      if name.endswith(".prog.tmp"):
        try:
          os.remove(os.path.join(parts_dir, name))
        except OSError:
          pass
  except OSError:
    pass
  base = os.path.basename(args.out)

  # Dynamic job list: many small batches so free workers can steal leftover games
  jobs: list[tuple[int, str, int, str]] = []  # (job_id, group, n_games, part_path)

  def _chunk_size(group: str, total: int, workers: int) -> int:
    # Larger chunks for fast R to reduce overhead, smaller for slow NN to balance well
    if total <= 0:
      return 0
    if group == "R":
      return total // workers#max(100, min(1000, max(1, total // max(1, workers))))
    if group == "NR":
      return total // workers#max(1, min(4, max(1, total // max(1, workers * 3))))
    return total // workers#max(1, min(2, max(1, total // max(1, workers * 6))))  # NN

  jid = 0

  def _add_group(group: str, total_games: int) -> None:
    nonlocal jid
    if total_games <= 0:
      return
    chunk = _chunk_size(group, total_games, args.workers)
    n_full, rem = divmod(total_games, chunk)
    for _ in range(n_full):
      part = os.path.join(parts_dir, f"{base}.{group}.job{jid:04d}.jsonl.gz")
      jobs.append((jid, group, chunk, part)); jid += 1
    if rem:
      part = os.path.join(parts_dir, f"{base}.{group}.job{jid:04d}.jsonl.gz")
      jobs.append((jid, group, rem, part)); jid += 1

  # Prioritize slow groups first so they start earlier
  _add_group("NN", nn_count)
  _add_group("NR", nr_count)
  _add_group("R", r_count)

  from concurrent.futures import ProcessPoolExecutor
  futures: list[Any] = []
  prog_map = {path + ".prog": group for (_, group, _, path) in jobs}

  with ProcessPoolExecutor(max_workers=args.workers) as ex:
    for (w_id, group, n, path_part) in jobs:
      futures.append(ex.submit(_worker_run_group, w_id, group, n, args.max_plies, args.seed, path_part, meta, args.negamax_movetime))

    last_print = 0.0
    # Poll progress files until all workers complete
    while True:
      if all(f.done() for f in futures):
        break
      now = time.time()
      if now - last_print < 10.0:
        time.sleep(1.0)
        continue
      last_print = now
      # Aggregate progress per group with fractional done
      # Aggregate progress per group with fractional done
      totals: dict[str, Any] = {
        "R": {"done": 0.0, "total": r_count, "workers": 0, "avg": [], "prior": _initial_sec_per_game("R", args.negamax_movetime, args.max_plies)},
        "NR": {"done": 0.0, "total": nr_count, "workers": 0, "avg": [], "prior": _initial_sec_per_game("NR", args.negamax_movetime, args.max_plies)},
        "NN": {"done": 0.0, "total": nn_count, "workers": 0, "avg": [], "prior": _initial_sec_per_game("NN", args.negamax_movetime, args.max_plies)},
      }
      for pf, grp in prog_map.items():
        pr = _read_progress(pf)
        if pr is None:
          continue
        totals[grp]["workers"] += 1
        totals[grp]["done"] += float(pr.get("done_float", pr.get("done", 0)))
        avg = pr.get("avg_sec_per_game")
        if isinstance(avg, (int, float)) and avg > 0:
          totals[grp]["avg"].append(float(avg))
        prior = pr.get("prior_sec_per_game")
        if isinstance(prior, (int, float)) and prior > 0:
          totals[grp]["prior"] = float(prior)

      # Make lines
      lines: list[str] = []
      grand_done = 0.0
      grand_total = 0
      etas: list[float] = []
      for tag in ("R", "NR", "NN"):
        d = totals[tag]["done"]
        t = totals[tag]["total"]
        w = totals[tag]["workers"]
        avg = (sum(totals[tag]["avg"]) / len(totals[tag]["avg"])) if totals[tag]["avg"] else None
        eta = _eta_group(d, t, w, avg, totals[tag]["prior"], tag, args.negamax_movetime, args.max_plies) if t else 0.0
        if t:
          lines.append(_progress_line2(tag, d, t, eta))
          grand_done += d
          grand_total += t
          etas.append(eta)

      if grand_total:
        all_eta = max(etas) if etas else 0.0  # parallel groups → max ETA
        lines.append(_progress_line2("ALL", grand_done, grand_total, all_eta))
      if lines:
        sys.stdout.write("\n" + "  |  ".join(lines) + " " * 8)
        sys.stdout.flush()

      time.sleep(1.0)

    # Ensure results fetched and print worker summaries
    for fut in futures:
      info = fut.result()
      print(f"\nworker {info['worker']} [{info['group']}] → {info['games']} games in {info['secs']:.1f}s")

  part_paths = [p for (_, _, _, p) in jobs]
  _concat_parts(part_paths, args.out)
  print(f"merged → {args.out}")
  # Merge error fragments
  cases = _merge_error_parts(part_paths, args.out)
  if cases:
    print(f"merged errors → {_error_final_path(args.out)} ({cases} cases)")
  # Cleanup parts and progress files
  parts_dir = args.out + ".parts"
  try:
    for name in os.listdir(parts_dir):
      try:
        os.remove(os.path.join(parts_dir, name))
      except OSError:
        pass
    os.rmdir(parts_dir)
  except OSError:
    pass


def main(argv: Optional[Iterable[str]] = None) -> int:
  args = parse_args(argv)

  if args.validate_path:
    errs = validate_jsonl(args.validate_path, sample=args.sample)
    return 1 if errs else 0

  if args.games <= 0 or not args.out:
    print("nothing to do: supply --games and --out or use --validate-path")
    return 2

  generate_parallel(args)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
