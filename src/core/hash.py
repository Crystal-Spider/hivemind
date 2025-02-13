import random
from typing import Final
from core.game import Position

_MAX_PIECES: Final[int] = 28
_BOARD_PAD_SIZE: Final[int] = 2
_MAX_BOARD_SIZE: Final[int] = (_MAX_PIECES + (_BOARD_PAD_SIZE * 2)) * 2 + 1
"""
Board size is at most as long as all pieces side by side.  
However, the hive is not necessarily centered with respect to the board origin (`Position(0, 0)`).  
To account for possible shifts, the board size is padded, with 1 extra cell to define a proper center.  
However, pieces might be placed all on one side, so we double the size (along with the padding).  
The pad size is small because, especially since the board size is doubled, it's very unlikely that a piece will actually get further than 32 tiles from the origin.

Another possible solution, safer but less efficient, would be to use a dynamic hash map for the board hashes rather than a fixed-size matrix.  
A smaller (e.g., 16x16) map could be initialized in the same way, and then progressively generate new hashes for new positions beyond the initial generation.

Lastly, the best solution would be to have a fixed-size matrix, but an efficient way to compute an hash invariant to offsets.
"""
_HALVED_BOARD_SIZE: Final[int] = _MAX_BOARD_SIZE // 2
_MAX_STACK_SIZE: Final[int] = 7
_HASH_SIZE: Final[int] = 64

def _rand() -> int:
  """
  Shortcut for `random.getrandbits(64)`.

  :return: Random 64-bit integer.
  :rtype: int
  """
  return random.getrandbits(_HASH_SIZE)

class ZobristHash:
  """
  Zobrist Hash invariant to offsets and 60° rotations.
  """
  random.seed(42)
  _HASH_PART_BY_TURN_COLOR: Final[int] = _rand()
  _HASH_PART_BY_LAST_MOVED_PIECE: Final[list[int]] = [_rand() for _ in range(_MAX_PIECES)]
  _HASH_PART_BY_POSITION: Final[list[list[list[list[int]]]]] = [[[[_rand() for _ in range(_MAX_STACK_SIZE)] for _ in range(_MAX_BOARD_SIZE)] for _ in range(_MAX_BOARD_SIZE)] for _ in range(_MAX_PIECES)]

  def __init__(self) -> None:
    self.value: int = 0

  def toggle_piece(self, piece_index: int, position: Position, stack: int) -> None:
    """
    Toggles the hash part for the specified piece at the specified position and stack.

    :param piece_index: Moved piece index.
    :type piece_index: int
    :param position: Moved piece position.
    :type position: Position
    :param stack: Moved piece elevation.
    :type stack: int
    """
    self.value ^= ZobristHash._HASH_PART_BY_POSITION[piece_index][position.q + _HALVED_BOARD_SIZE][position.r + _HALVED_BOARD_SIZE][stack]

  def toggle_last_moved_piece(self, piece_index: int) -> None:
    """
    Toggles the hash part for the last moved piece, specified by its index.

    :param piece_index: Moved piece index.
    :type piece_index: int
    """
    self.value ^= ZobristHash._HASH_PART_BY_LAST_MOVED_PIECE[piece_index]

  def toggle_turn(self) -> None:
    """
    Toggles the hash part for the turn color.
    """
    self.value ^= ZobristHash._HASH_PART_BY_TURN_COLOR
