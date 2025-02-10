from typing import Optional
from enum import Flag, auto

class TranspositionTableEntryType(Flag):
  EXACT = auto()
  LOWER_BOUND = auto()
  UPPER_BOUND = auto()

class TranspositionTableEntry:
  def __init__(self, type: TranspositionTableEntryType, value: float, depth: int, best_move: str) -> None:
    self.type: TranspositionTableEntryType = type
    self.value: float = value
    self.depth: int = depth
    self.best_move: str = best_move

class TranspositionTable:
  def __init__(self) -> None:
    self._table: dict[int, Optional[TranspositionTableEntry]] = {}

  def __getitem__(self, key: int) -> Optional[TranspositionTableEntry]:
    return self._table[key]

  def __setitem__(self, key: int, entry: TranspositionTableEntry) -> None:
    if (current := self._table[key]) is None or entry.depth > current.depth:
      self._table[key] = entry
