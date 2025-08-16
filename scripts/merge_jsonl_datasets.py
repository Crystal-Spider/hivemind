#!/usr/bin/env python3
# 2-space indentation, strict typing
import argparse
import gzip
import io
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Any

"""
Merge many selfplay JSONL(.gz) files into a single dataset, keeping the
same schema produced by scripts/selfplay_to_jsonl.py.

Features
- Accepts a mix of .jsonl and .jsonl.gz inputs
- Streams and concatenates without loading everything in memory
- Optional variant filter (keep only Base+MLP)
- Merges per-input error files (<in>-errors.txt) into one (<out>-errors.txt)
- Optional shuffle-by-chunks to lightly interleave inputs (stable enough for training)

Usage
  python scripts/merge_jsonl_datasets.py \
    --out data/merged.jsonl.gz data/shard1.jsonl.gz data/shard2.jsonl.gz ...

  # filter to Base+MLP only (by header in game_string)
  python scripts/merge_jsonl_datasets.py --variant Base+MLP \
    --out data/merged.jsonl.gz data/*.jsonl.gz

  # light interleave (chunked), default off
  python scripts/merge_jsonl_datasets.py --shuffle-chunk 1000 \
    --out data/merged.jsonl.gz data/*.jsonl.gz
"""

# ------------------------------ CLI -----------------------------------
class Args(argparse.Namespace):
  out: str
  inputs: list[str]
  variant: Optional[str]
  shuffle_chunk: int


def parse_args(argv: Optional[Iterable[str]] = None) -> Args:
  p = argparse.ArgumentParser(description="Merge selfplay JSONL(.gz) datasets")
  p.add_argument("--variant", default=None, help="keep only games whose game_string header contains this exact token (e.g., 'Base+MLP')")
  p.add_argument("--shuffle-chunk", type=int, default=0, help="if >0, read about this many records at a time from each input in round-robin to lightly interleave")
  p.add_argument("--out", required=True, help="output .jsonl(.gz)")
  p.add_argument("inputs", nargs="+", help="input .jsonl(.gz) files")
  return p.parse_args(list(argv) if argv is not None else None)  # type: ignore[return-value]


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


def _error_final_path(out_path: Path) -> Path:
  name = out_path.name
  for ext in (".gz", ".jsonl", ".json"):
    if name.endswith(ext):
      name = name[: -len(ext)]
  return out_path.with_name(f"{name}-errors.txt")


# ------------------------------ Merge ----------------------------------

def _pass_variant(rec: dict[str, Any], variant: Optional[str]) -> bool:
  if not variant:
    return True
  gs = rec.get("game_string")
  if not isinstance(gs, str):
    return False
  # fast check: header is before first ';'
  head = gs.split(";", 1)[0].strip()
  return head == variant or variant in head


def merge(out_path: Path, inputs: Sequence[Path], variant: Optional[str], shuffle_chunk: int) -> int:
  n_in = 0
  n_out = 0
  # writer
  with _open_writer(out_path) as out:
    # simple path: no chunk interleave
    if shuffle_chunk <= 0:
      for ip in inputs:
        with _open_reader(ip) as rd:
          for line in rd:
            if not line.strip():
              continue
            n_in += 1
            try:
              rec = json.loads(line)
            except Exception:
              continue
            if not _pass_variant(rec, variant):
              continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    else:
      # round-robin chunk read
      readers = [(_open_reader(ip), ip) for ip in inputs]
      try:
        exhausted = [False] * len(readers)
        while not all(exhausted):
          for i, (rd, ip) in enumerate(readers):
            if exhausted[i]:
              continue
            taken = 0
            while taken < shuffle_chunk:
              line = rd.readline()
              if not line:
                exhausted[i] = True
                break
              if not line.strip():
                continue
              n_in += 1
              try:
                rec = json.loads(line)
              except Exception:
                continue
              if not _pass_variant(rec, variant):
                continue
              out.write(json.dumps(rec, ensure_ascii=False) + "\n")
              n_out += 1
              taken += 1
      finally:
        for rd, _ in readers:
          try:
            rd.close()
          except Exception:
            pass

  # merge per-input errors into <out>-errors.txt
  err_out = _error_final_path(out_path)
  total_err = 0
  with open(err_out, "w", encoding="utf-8") as ef:
    for ip in inputs:
      ep = Path(str(ip) + ".errors")
      if ep.exists():
        try:
          data = ep.read_text(encoding="utf-8")
          if data:
            ef.write(data)
            total_err += data.count("\n\n")
        except OSError:
          pass
  if total_err == 0:
    try:
      os.remove(err_out)
    except OSError:
      pass

  print(f"merged: {n_out}/{n_in} → {out_path}")
  return 0 if n_out > 0 else 2


# ------------------------------ Main -----------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
  a = parse_args(argv)
  outp = Path(a.out).resolve()
  inputs = [Path(s).resolve() for s in a.inputs]
  # existence check
  ok = True
  for ip in inputs:
    if not ip.exists():
      print(f"missing: {ip}")
      ok = False
  if not ok:
    return 2
  return merge(outp, inputs, a.variant, a.shuffle_chunk)


if __name__ == "__main__":
  raise SystemExit(main())
