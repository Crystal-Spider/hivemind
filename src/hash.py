import random
from typing import Final, Callable, Optional
from enum import StrEnum
from game import Position

class ZobristHashReference(StrEnum):
  """
  Reference points for the Zobrist Hash.
  """
  ORIGIN = "origin"
  """
  Position of the piece used as the invariant origin for the board.
  """
  ORIENTATION = "orientation"
  """
  Position of the piece used as the invariant for the board orientation.
  """

class ZobristHash:
  """
  Zobrist Hash invariant to offsets and 60° rotations.
  """
  EMPTY_BOARD: Final[int] = 0
  """
  Hash value for an empty board.
  """

  def __init__(self, num_pieces: int, board_size: int, board_stack_size: int, references: Callable[[ZobristHashReference], Optional[Position]]) -> None:
    """
    Create a new Zobrist Hash instance invariant to offsets and 60° rotations.

    :param num_pieces: How many pieces are there in the game.
    :type num_pieces: int
    :param board_size: How big the board could ever get.
    :type board_size: int
    :param board_stack_size: How high a stack of pieces could ever get.
    :type board_stack_size: int
    :param references:  
      Function that must return the position of the specified reference piece.  
      The reference position is allowed to be None, meaning the current state of the board is invariant to any transformation based on that reference.  
      Explicitly: when ORIGIN is None, the board is invariant to offsets (is empty); when ORIENTATION is None, the board is invariant to rotations (has only one piece).
    :type reference: Callable[[ZobristHashReference], Optional[Position]]
    """
    self.value: int = self.EMPTY_BOARD
    self.references: Callable[[ZobristHashReference], Optional[Position]] = references
    self.board_size: int = board_size
    self._hash_part_by_turn_color: int = ZobristHash.rand()
    self._hash_part_by_last_moved_piece: list[int] = [ZobristHash.rand() for _ in range(num_pieces)]
    self._hash_part_by_position = [[[[ZobristHash.rand() for _ in range(board_stack_size)] for _ in range(board_size)] for _ in range(board_size)] for _ in range(num_pieces)]

  @staticmethod
  def rand() -> int:
    """
    Shortcut for `random.getrandbits(64)`.

    :return: Random 64-bit integer.
    :rtype: int
    """
    return random.getrandbits(64)

  def _canonical_offset(self, position: Position) -> Position:
    """
    Offset the position by the reference to get the canonical position.

    :param position: Original position.
    :type position: Position
    :return: Position with the canonical offset.
    :rtype: Position
    """
    return position - (self.references(ZobristHashReference.ORIGIN) or Position(0, 0))

  def _canonical_orientation(self, position: Position) -> Position:
    """
    Rotate the position (which must be offset first) until the orientation is in the canonical quadrant.

    :param position: Original position (with canonical offset)
    :type position: Position
    :return: Position with the canonical orientation.
    :rtype: Position
    """
    orientation = self.references(ZobristHashReference.ORIENTATION)
    rotated = position
    print(f"orientation: {orientation}, rotated: {rotated}")
    # If the orientation is None, there is no second piece on the board yet, so there is no orientation to consider.
    # If the orientation is not None, rotate until the orientation moves into the "quadrant" where both q and r are positive (canonical orientation).
    while orientation is not None and not (orientation.q > 0 and orientation.r >= 0):
      orientation = orientation.anticlockwise()
      rotated = rotated.anticlockwise()
      print(f"orientation: {orientation}, rotated: {rotated}")
    print(f"orientation: {orientation}, rotated: {rotated}")
    return rotated
  
  def _canonical_position(self, position: Position) -> Position:
    """
    Transform the position into its canonical form.

    :param position: Original position.
    :type position: Position
    :return: Canonical position.
    :rtype: Position
    """
    return self._canonical_orientation(self._canonical_offset(position))

  def _canonical_hash(self, piece_index: int, position: Position, stack: int) -> int:
    canonical_position = self._canonical_position(position)
    return self._hash_part_by_position[piece_index][canonical_position.q + self.board_size // 2][canonical_position.r + self.board_size // 2][stack]

  def toggle_piece(self, piece_index: int, position: Position, stack: int) -> None:
    self.value ^= self._canonical_hash(piece_index, position, stack)

  def toggle_last_moved_piece(self, piece_index: int) -> None:
    self.value ^= self._hash_part_by_last_moved_piece[piece_index]

  def toggle_turn(self) -> None:
    self.value ^= self._hash_part_by_turn_color
