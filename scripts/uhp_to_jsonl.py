#!/usr/bin/env python3
# 2-space indentation, strict typing
import argparse
import gzip
import io
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Any

"""
Convert a TXT file of UHP GameStrings (one per line) into a JSONL(.gz)
with the same schema used by scripts/selfplay_to_jsonl.py:

  {
    "game_string": str,           # canonicalized by the engine
    "moves": [str, ...],          # UHP move strings played in order (including 'pass')
    "result": "W"|"B"|"D"|null,   # from side to move perspective
    "ply_count": int,
    "meta": {"source": "uhp_txt_to_jsonl", "line_no": int}
  }

Usage:
  python scripts/uhp_txt_to_jsonl.py --in data/games.txt --out data/selfplay.jsonl.gz

Notes:
- Lines that fail to parse/play are logged into <out>-errors.txt
  (each error: the original line, then the exception, blank line).
- Blank/comment lines (#...) are ignored.
"""

# ----------------- Import engine from ./src -----------------
ROOT: Path = Path(__file__).resolve().parents[1]
src_path = str(ROOT / "src")
if src_path not in sys.path:
  sys.path.insert(0, src_path)

from core.board import Board  # type: ignore
from core.enums import GameState  # type: ignore


# --------------------------- Helpers ---------------------------

def _split_uhp_gamestring(gs: str) -> list[str]:
  """Split a UHP GameString into segments on *unescaped* semicolons.
  Keeps backslashes in tokens. Trims whitespace; drops empties.
  Example: "a\\;b;c" → ["a\\;b", "c"].
  """
  parts: list[str] = []
  cur: list[str] = []
  esc = False
  for ch in gs:
    if esc:
      # keep escaped char verbatim
      cur.append(ch)
      esc = False
      continue
    if ch == "\\":
      cur.append(ch)
      esc = True
      continue
    if ch == ";":
      token = "".join(cur).strip()
      if token:
        parts.append(token)
      cur = []
      continue
    cur.append(ch)
  token = "".join(cur).strip()
  if token:
    parts.append(token)
  return parts


def _map_result(state: GameState) -> Optional[str]:
  if state is GameState.DRAW:
    return "D"
  if state is GameState.WHITE_WINS:
    return "W"
  if state is GameState.BLACK_WINS:
    return "B"
  return None


# ---------------------------- I/O --------------------------------
class JsonlWriter:
  def __init__(self, path: Path) -> None:
    self.path = path
    if str(path).endswith(".gz"):
      self._fh: io.TextIOBase = io.TextIOWrapper(gzip.open(path, "wb"), encoding="utf-8")
    else:
      self._fh = open(path, "w", encoding="utf-8")

  def write(self, obj: dict[Any, Any]) -> None:
    self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

  def close(self) -> None:
    self._fh.close()


# ---------------------------- Main --------------------------------
class Args(argparse.Namespace):
  inp: str
  out: Optional[str]
  errors: Optional[str]


def parse_args(argv: Optional[Iterable[str]] = None) -> Args:
  p = argparse.ArgumentParser(description="Convert TXT (UHP GameStrings) → JSONL(.gz)")
  p.add_argument("--in", dest="inp", required=True, help="input .txt with one GameString per line")
  p.add_argument("--out", default=None, help="output .jsonl(.gz); default = input name with .jsonl.gz")
  p.add_argument("--errors", default=None, help="errors output .txt; default = <out>-errors.txt")
  return p.parse_args(list(argv) if argv is not None else None)  # type: ignore[return-value]


def _default_out(inp: Path) -> Path:
  name = inp.name
  if name.endswith(".txt"):
    name = name[:-4]
  return inp.with_name(name + ".jsonl.gz")


def _error_path(out_path: Path, explicit: Optional[str]) -> Path:
  if explicit:
    return Path(explicit)
  name = out_path.name
  for ext in (".gz", ".jsonl", ".json"):
    if name.endswith(ext):
      name = name[: -len(ext)]
  return out_path.with_name(f"{name}-errors.txt")


def convert(inp: Path, out_path: Path, err_path: Path) -> int:
  total = 0
  ok = 0
  err_fh: Optional[io.TextIOWrapper] = None
  writer = JsonlWriter(out_path)

  try:
    err_fh = open(err_path, "w", encoding="utf-8")
  except OSError:
    err_fh = None

  with open(inp, "r", encoding="utf-8", errors="strict") as f:
    for ln, line in enumerate(f, start=1):
      s = line.strip()
      if not s or s.startswith("#"):
        continue
      total += 1
      try:
        # Build moves by playing segments that are actual moves.
        b = Board()
        moves: list[str] = []
        for seg in _split_uhp_gamestring(s):
          try:
            b.play(seg)
            moves.append(seg)
          except Exception:
            # Not a move segment (headers like Base, Draw, White[...]) → ignore
            continue
        # Canonicalize GameString and result
        canon = str(b)
        rec: dict[str, Any]= {
          "game_string": canon,
          "moves": moves,
          "result": _map_result(b.state),
          "ply_count": len(moves),
          "meta": {"source": "uhp_txt_to_jsonl", "line_no": ln},
        }
        writer.write(rec)
        ok += 1
      except Exception as e:
        if err_fh is not None:
          try:
            err_fh.write(s + "\n" + repr(e) + "\n\n")
          except OSError:
            pass
        # continue with next line
        continue

  writer.close()
  if err_fh is not None:
    err_fh.close()

  print(f"converted {ok}/{total} records → {out_path}")
  if ok == 0 and total > 0:
    return 2
  return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
  args = parse_args(argv)
  inp = Path(args.inp).resolve()
  if not inp.exists():
    print(f"input not found: {inp}")
    return 2
  outp = Path(args.out).resolve() if args.out else _default_out(inp)
  outp.parent.mkdir(parents=True, exist_ok=True)
  errp = _error_path(outp, args.errors)
  rc = convert(inp, outp, errp)
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
