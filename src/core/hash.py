import random
from typing import Final
from core.game import Position

class ZobristHash:
  """
  Zobrist Hash invariant to offsets and 60° rotations.
  """
  EMPTY_BOARD: Final[int] = 0
  """
  Hash value for an empty board.
  """

  def __init__(self, num_pieces: int, board_size: int, board_stack_size: int) -> None:
    """
    Create a new Zobrist Hash instance invariant to offsets and 60° rotations.

    :param num_pieces: How many pieces are there in the game.
    :type num_pieces: int
    :param board_size: How big the board could ever get.
    :type board_size: int
    :param board_stack_size: How high a stack of pieces could ever get.
    :type board_stack_size: int
    :type reference: Callable[[ZobristHashReference], Optional[Position]]
    """
    self.value: int = self.EMPTY_BOARD
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
    self.value ^= self._hash_part_by_position[piece_index][position.q + self.board_size // 2][position.r + self.board_size // 2][stack]

  def toggle_last_moved_piece(self, piece_index: int) -> None:
    """
    Toggles the hash part for the last moved piece, specified by its index.

    :param piece_index: Moved piece index.
    :type piece_index: int
    """
    self.value ^= self._hash_part_by_last_moved_piece[piece_index]

  def toggle_turn(self) -> None:
    """
    Toggles the hash part for the turn color.
    """
    self.value ^= self._hash_part_by_turn_color
