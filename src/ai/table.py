from collections import OrderedDict
from typing import Optional, Final, Generic, TypeVar, Self
from abc import ABC, abstractmethod
from enum import IntEnum, auto

class AgingTableEntry(ABC):
  """
  Table entry for tables that implement deletion of entries based on age.
  """

  def __init__(self) -> None:
    self.age: int = 0

_Entry = TypeVar("_Entry", bound=AgingTableEntry)
"""
Entry type generic for `AgingTable`.
"""
_Value = TypeVar("_Value")
"""
Value type generic for `AgingTable`.
"""

class AgingTable(ABC, Generic[_Entry, _Value]):
  """
  Aging table, storing `_Entry` type entries into a hash table that supports deleting them after their are not accessed for a long time (they aged).  
  The table might optionally wrap the inner entries and expose a value-based interface instead, depending on whether the `_Value` type is equal to the `_Entry` type.
  """

  def __init__(self, max_age: int, max_size: int) -> None:
    self._table: OrderedDict[int, _Entry] = OrderedDict()
    self._max_age: Final[int] = max_age
    self._max_size: Final[int] = max_size

  def __setitem__(self, key: int, value: _Value) -> None:
    if len(self._table) > self._max_size:
      self._table.popitem(last=False)
    self._table[key] = self._entry_from_value(value)
    self._table.move_to_end(key)

  def __delitem__(self, key: int) -> None:
    del self._table[key]

  def __getitem__(self, key: int) -> Optional[_Value]:
    return self._value_from_entry(self._table[key]) if key in self._table else None

  def __len__(self) -> int:
    return len(self._table)

  def __contains__(self, key: object) -> bool:
    return key in self._table

  def __iter__(self):
    return iter(self._table)

  def __eq__(self, other: object):
    return isinstance(other, type(self)) and self._table == other._table

  @abstractmethod
  def copy(self) -> Self:
    """
    Returns a (possibly shallow) copy of this instance.
    """

  @abstractmethod
  def _entry_from_value(self, value: _Value) -> _Entry:
    """
    Maps a value with its corresponding entry.

    :param value: Entry value.
    :type value: _Value
    :return: Entry.
    :rtype: _Entry
    """

  @abstractmethod
  def _value_from_entry(self, entry: _Entry) -> _Value:
    """
    Maps an entry with its corresponding value.

    :param entry: Entry.
    :type entry: _Entry
    :return: Entry value.
    :rtype: _Value
    """

  def keys(self):
    """
    Returns all stored keys.

    :return: A set-like object providing a view on the stored keys.
    :rtype: OrderedDictKeysView
    """
    return self._table.keys()

  def values(self):
    """
    Returns all stored values.

    :return: A set-like object providing a view on the stored values.
    :rtype: OrderedDictValuesView
    """
    return self._table.values()

  def items(self):
    """
    Returns all stored items.

    :return: A set-like object providing a view on the stored items.
    :rtype: OrderedDictItemsView
    """
    return self._table.items()

  def popitem(self, last: bool = True) -> tuple[int, _Value]:
    """
    Removes and returns a (key, value) pair from the storage.

    :param last: Whether to return pairs in LIFO (`True`) or FIFO (`False`) order, defaults to `True`.
    :type last: bool, optional
    :return: _description_
    :rtype: tuple[int, _Value]
    """
    key, entry = self._table.popitem(last)
    return key, self._value_from_entry(entry)

  def clear(self) -> None:
    """
    Clears the storage from all entries.
    """
    self._table.clear()

  def flush(self) -> None:
    """
    Ages entries and removes old ones.
    """
    to_delete = [key for key, entry in self._table.items() if entry.age >= self._max_age]
    for key in to_delete:
      del self._table[key]
    for key, entry in self._table.items():
      entry.age += 1

class TranspositionTableEntryType(IntEnum):
  """
  Alpha-beta pruning node evaluation type.
  """
  EXACT = auto()
  """
  Exact evaluation was performed for the node.
  """
  LOWER_BOUND = auto()
  """
  The node is a lower bound, the search failed high.
  """
  UPPER_BOUND = auto()
  """
  The node is an upper bound, the search failed low.
  """

class TranspositionTableEntry(AgingTableEntry):
  """
  Transposition table entry.
  """

  def __init__(self, type: TranspositionTableEntryType, value: float, depth: int, move: Optional[str]) -> None:
    super().__init__()
    self.type: TranspositionTableEntryType = type
    self.value: float = value
    self.depth: int = depth
    self.move: Optional[str] = move

class TranspositionTable(AgingTable[TranspositionTableEntry, TranspositionTableEntry]):
  """
  Transposition table with max size and entry aging.
  """

  def __init__(self, max_age: int = 10, max_size: int = 10_000_000) -> None:
    super().__init__(max_age, max_size)

  def __setitem__(self, key: int, value: TranspositionTableEntry) -> None:
    if key not in self._table or value.depth > self._table[key].depth:
      super().__setitem__(key, value)

  def __getitem__(self, key: int) -> Optional[TranspositionTableEntry]:
    return self._table[key] if key in self._table else None

  def _entry_from_value(self, value: TranspositionTableEntry) -> TranspositionTableEntry:
    return value

  def _value_from_entry(self, entry: TranspositionTableEntry) -> TranspositionTableEntry:
    return entry

  def copy(self):
    table = TranspositionTable(self._max_age, self._max_size)
    table._table = self._table.copy()
    return table

class ScoreTableEntry(AgingTableEntry):
  """
  Score table entry.
  """

  def __init__(self, score: float) -> None:
    super().__init__()
    self.score: float = score

class ScoreTable(AgingTable[ScoreTableEntry, float]):
  """
  Score cache table with max size and entry aging.
  """

  def __init__(self, max_age: int = 10, max_size: int = 10_000_000) -> None:
    super().__init__(max_age, max_size)

  def __setitem__(self, key: int, value: float) -> None:
    if key not in self._table:
      super().__setitem__(key, value)

  def _entry_from_value(self, value: float) -> ScoreTableEntry:
    return ScoreTableEntry(value)

  def _value_from_entry(self, entry: ScoreTableEntry) -> float:
    return entry.score

  def copy(self):
    table = ScoreTable(self._max_age, self._max_size)
    table._table = self._table.copy()
    return table
