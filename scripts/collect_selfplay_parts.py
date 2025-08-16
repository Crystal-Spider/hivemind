#!/usr/bin/env python3
# Strict typing, 2-space indentation
import argparse
import gzip
from pathlib import Path
from typing import Iterable, Optional, Sequence

"""
Collect leftover self-play parts (from a crashed or interrupted run) and
produce the final JSONL(.gz) + merged error file.

It expects the generator's layout:
  <OUT>.parts/
    <base>.<GROUP>.job####.jsonl.gz
    <same>.prog / .prog.tmp (ignored)
    <same>.errors (optional per-part)

Usage examples:
  # infer --out from --parts-dir (strip trailing .parts)
  python scripts/collect_selfplay_parts.py --parts-dir data/selfplay.jsonl.gz.parts

  # explicit
  python scripts/collect_selfplay_parts.py --parts-dir data/run123.parts --out data/selfplay.jsonl.gz

  # clean up parts after merge
  python scripts/collect_selfplay_parts.py --parts-dir data/selfplay.jsonl.gz.parts --cleanup
"""

# ------------------------------- CLI -----------------------------------
class Args(argparse.Namespace):
  parts_dir: str
  out: Optional[str]
  cleanup: bool


def parse_args(argv: Optional[Iterable[str]] = None) -> Args:
  p = argparse.ArgumentParser(description="Collect leftover self-play parts and merge into a single JSONL(.gz)")
  p.add_argument("--parts-dir", default="data/selfplay.jsonl.gz.parts", help="directory holding <out>.parts")
  p.add_argument("--out", default=None, help="final JSONL(.gz); default = <parts-dir> without trailing .parts")
  p.add_argument("--cleanup", action="store_true", help="remove parts dir after successful merge")
  return p.parse_args(list(argv) if argv is not None else None)  # type: ignore[return-value]


# ---------------------------- Utilities --------------------------------

def _is_gz(path: Path) -> bool:
  return path.suffix == ".gz" or path.name.endswith(".jsonl.gz")


def _error_final_path(out_path: Path) -> Path:
  name = out_path.name
  for ext in (".gz", ".jsonl", ".json"):
    if name.endswith(ext):
      name = name[: -len(ext)]
  return out_path.with_name(f"{name}-errors.txt")


def _list_parts(parts_dir: Path) -> list[Path]:
  parts: list[Path] = []
  for p in sorted(parts_dir.iterdir()):
    if not p.is_file():
      continue
    n = p.name
    if n.endswith(".prog") or n.endswith(".prog.tmp") or n.endswith(".errors"):
      continue
    if n.endswith(".jsonl") or n.endswith(".jsonl.gz"):
      parts.append(p)
  return parts


def _concat(parts: Sequence[Path], out_path: Path) -> None:
  if not parts:
    raise SystemExit(f"no part files found in {out_path}.parts (nothing to merge)")

  # Create parent
  out_path.parent.mkdir(parents=True, exist_ok=True)
  # Truncate destination
  if out_path.exists():
    out_path.unlink()

  if _is_gz(out_path):
    # Mixed inputs allowed. Gz parts are copied raw (member concatenation); plain parts are re-compressed and appended.
    for part in parts:
      if _is_gz(part):
        with out_path.open("ab") as outb, part.open("rb") as f:
          outb.write(f.read())
      else:
        with gzip.open(out_path, "ab") as gout, part.open("rt", encoding="utf-8") as tin:
          for line in tin:
            gout.write(line.encode("utf-8"))
  else:
    # Plain output. Decompress gz parts on the fly.
    with out_path.open("wt", encoding="utf-8") as outt:
      for part in parts:
        if _is_gz(part):
          with gzip.open(part, "rt", encoding="utf-8", errors="strict") as tin:
            for line in tin:
              outt.write(line)
        else:
          with part.open("rt", encoding="utf-8") as tin:
            for line in tin:
              outt.write(line)


def _merge_errors(parts: Sequence[Path], out_path: Path) -> int:
  err_out = _error_final_path(out_path)
  err_out.parent.mkdir(parents=True, exist_ok=True)
  total_cases = 0
  with err_out.open("wt", encoding="utf-8") as out:
    for p in parts:
      ep = Path(str(p) + ".errors")
      if not ep.exists():
        continue
      try:
        data = ep.read_text(encoding="utf-8")
      except OSError:
        continue
      if data:
        out.write(data)
        total_cases += data.count("\n\n")
  if total_cases == 0:
    try:
      err_out.unlink()
    except OSError:
      pass
  return total_cases


# ------------------------------ Main -----------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
  args = parse_args(argv)
  parts_dir = Path(args.parts_dir).resolve()
  if not parts_dir.exists() or not parts_dir.is_dir():
    print(f"parts dir not found: {parts_dir}")
    return 2

  # Infer out path
  out_path = Path(args.out).resolve() if args.out else parts_dir.with_name(parts_dir.name[:-6]) if parts_dir.name.endswith(".parts") else parts_dir.with_suffix("")

  parts = _list_parts(parts_dir)
  if not parts:
    print(f"no parts to merge in {parts_dir}")
    return 3

  # Stable path sort is enough (names include job####)
  parts.sort()

  print(f"merging {len(parts)} parts → {out_path}")
  _concat(parts, out_path)
  print(f"merged → {out_path}")

  cases = _merge_errors(parts, out_path)
  if cases:
    print(f"merged errors → {_error_final_path(out_path)} ({cases} cases)")

  if args.cleanup:
    try:
      for p in parts_dir.iterdir():
        try:
          p.unlink()
        except OSError:
          pass
      parts_dir.rmdir()
      print(f"cleaned up {parts_dir}")
    except OSError:
      pass

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
