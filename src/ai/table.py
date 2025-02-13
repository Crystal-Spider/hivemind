from collections import OrderedDict
from typing import Optional, Final
from enum import IntEnum, auto

class TranspositionTableEntryType(IntEnum):
  EXACT = auto()
  LOWER_BOUND = auto()
  UPPER_BOUND = auto()

class TranspositionTableEntry:
  def __init__(self, type: TranspositionTableEntryType, value: float, depth: int, move: Optional[str]) -> None:
    self.type: TranspositionTableEntryType = type
    self.value: float = value
    self.depth: int = depth
    self.move: Optional[str] = move
    self.age: int = 0

class TranspositionTable():
  def __init__(self, max_age: int = 10, max_size: int = 10_000_000) -> None:
    self._table: OrderedDict[int, TranspositionTableEntry] = OrderedDict()
    self._max_age: Final[int] = max_age
    self._max_size: Final[int] = max_size

  def __setitem__(self, key: int, entry: TranspositionTableEntry) -> None:
    if key not in self._table or entry.depth > self._table[key].depth:
      if len(self._table) > self._max_size:
        self._table.popitem(last=False)
      self._table[key] = entry
      self._table.move_to_end(key)

  def __delitem__(self, key: int) -> None:
    del self._table[key]

  def __getitem__(self, key: int) -> Optional[TranspositionTableEntry]:
    return self._table[key] if key in self._table else None

  def __len__(self) -> int:
    return len(self._table)

  def __contains__(self, key: object) -> bool:
    return key in self._table

  def __iter__(self):
    return iter(self._table)

  def __cmp__(self, other: object):
    return isinstance(other, TranspositionTable) and self._table == other._table

  def keys(self):
    return self._table.keys()

  def values(self):
    return self._table.values()

  def items(self):
    return self._table.items()

  def popitem(self, last: bool = True) -> tuple[int, TranspositionTableEntry]:
    return self._table.popitem(last)

  def copy(self):
    table = TranspositionTable(self._max_age, self._max_size)
    table._table = self._table.copy()
    return table

  def clear(self) -> None:
    self._table.clear()

  def flush(self) -> None:
    to_delete = [key for key, entry in self._table.items() if entry.age >= self._max_age]
    for key in to_delete:
      del self._table[key]
    for key, entry in self._table.items():
      entry.age += 1
