from typing import Optional
from random import choice
from time import sleep, time
from abc import ABC, abstractmethod
from core.board import Board
from core.game import Move
from core.enums import GameState
from ai.table import TranspositionTable, TranspositionTableEntry, TranspositionTableEntryType, ScoreTable

class Brain(ABC):
  """
  Base abstract class for AI agents.
  """

  def __init__(self) -> None:
    self._best_move_cache: Optional[str] = None
    self._last_max_depth: int = 0
    self._last_time_limit: int = 0
    self._last_hash: int = 0

  def find_best_move(self, board: Board, max_branching_factor: int, max_depth: int = 0, time_limit: int = 0) -> str:
    """
    Finds the best move for the given board state, following the agent's policy.

    :param board: Current playing board.
    :type board: Board
    :param max_branching_factor: Maximum branching factor.
    :type max_branching_factor: int
    :param max_depth: Maximum lookahead depth, defaults to `0`.
    :type max_depth: int, optional
    :param time_limit: Maximum time (in seconds) to calculate the best move, defaults to `0`.
    :type time_limit: int, optional
    :return: Stringified best move.
    :rtype: str
    """
    if not self._best_move_cache or self._last_hash != board.hash() or self._last_max_depth != max_depth or self._last_time_limit != time_limit:
      self._empty_cache()
      self._last_max_depth = max_depth
      self._last_time_limit = time_limit
      self._last_hash = board.hash()
      self._best_move_cache = self._find_best_move(board, max_branching_factor, max_depth, time_limit)
    return self._best_move_cache

  @abstractmethod
  def _find_best_move(self, board: Board, max_branching_factor: int, max_depth: int = 0, time_limit: int = 0) -> str:
    """
    Finds the best move according to this agent's strategy.

    :param board: Current playing board.
    :type board: Board
    :param max_branching_factor: Maximum branching factor.
    :type max_branching_factor: int
    :param max_depth: Maximum lookahead depth, defaults to `0`.
    :type max_depth: int, optional
    :param time_limit: Maximum time (in seconds) to calculate the best move, defaults to `0`.
    :type time_limit: int, optional
    :return: Stringified best move.
    :rtype: str
    """

  def _empty_cache(self) -> None:
    """
    | Empties the current cache for the best move.
    | Might empty more data depending on the agent.
    """
    self._best_move_cache = None

class Random(Brain):
  """
  Random acting AI agent.
  """

  def _find_best_move(self, board: Board, max_branching_factor: int, max_depth: int = 0, time_limit: int = 0) -> str:
    sleep(0.5)
    return choice(board.valid_moves.split(";"))

class AlphaBetaPruner(Brain):
  """
  AI agent following an alpha-beta pruning policy.
  """

  def __init__(self) -> None:
    super().__init__()
    self._transpos_table: TranspositionTable = TranspositionTable()
    self._pv_table: dict[int, Move] = {}
    self._killer_moves: dict[int, list[Move]] = {}
    self._history_heuristic: dict[Move, int] = {}
    self._cached_scores: ScoreTable = ScoreTable()
    self._visited_nodes: int = 0
    self._cutoffs: int = 0

  def _find_best_move(self, board: Board, max_branching_factor: int, max_depth: int = 0, time_limit: int = 0) -> str:
    start_time = time()
    best_move = None
    scores: list[tuple[Optional[Move], float]] = []
    depth = 0
    try:
      while not max_depth or depth < max_depth:
        depth += 1
        best_move, score = self._alpha_beta_search(board, max_branching_factor, depth, float('-inf'), float('inf'), start_time, time_limit)
        scores.append((best_move, score))
        if time_limit and time() - start_time > time_limit:
          break
    except TimeoutError:
      pass
    self._transpos_table.flush()
    print(f"Visited nodes: {self._visited_nodes}; Cutoffs: {self._cutoffs}; Scores: {scores}; Time: {time() - start_time}")
    self._visited_nodes = 0
    self._cutoffs = 0
    return board.stringify_move(best_move)

  def _alpha_beta_search(self, board: Board, max_branching_factor: int, depth: int, alpha: float, beta: float, start_time: float, time_limit: int) -> tuple[Optional[Move], float]:
    if time_limit and time() - start_time > time_limit:
      raise TimeoutError("Time limit exceeded during alpha-beta pruning search")

    self._visited_nodes += 1
    node_hash = board.hash()
    cached_entry = self._transpos_table[node_hash]
    if cached_entry and cached_entry.depth >= depth:
      match cached_entry.type:
        case TranspositionTableEntryType.EXACT:
          self._cutoffs += 1
          return cached_entry.move, cached_entry.value
        case TranspositionTableEntryType.LOWER_BOUND:
          alpha = max(alpha, cached_entry.value)
        case TranspositionTableEntryType.UPPER_BOUND:
          beta = min(beta, cached_entry.value)
      if alpha >= beta:
        self._cutoffs += 1
        return cached_entry.move, cached_entry.value

    if depth == 0 or board.gameover:
      return None, self._evaluate(board, None)

    best_move = self._pv_table.get(node_hash, None)
    moves = list(board.calculate_valid_moves())
    moves.sort(key=lambda m: self._move_order_heuristic(board, m, best_move, depth), reverse=True)
    if len(moves) > max_branching_factor:
      del moves[max_branching_factor:]

    best_value = float('-inf')
    for move in moves:
      board.play_parsed(move)
      _, value = self._alpha_beta_search(board, max_branching_factor, depth - 1, -beta, -alpha, start_time, time_limit)
      board.undo()
      value = -value
      if value > best_value:
        best_value = value
        best_move = move
      alpha = max(alpha, best_value)
      if alpha >= beta:
        self._cutoffs += 1
        self._store_killer_move(depth, move)
        break
    entry_type = TranspositionTableEntryType.EXACT if best_value < beta else TranspositionTableEntryType.LOWER_BOUND if best_value > alpha else TranspositionTableEntryType.UPPER_BOUND
    self._transpos_table[node_hash] = TranspositionTableEntry(entry_type, best_value, depth, best_move)
    if best_move:
      self._pv_table[node_hash] = best_move
      self._update_history_heuristic(best_move, depth)
    return best_move, best_value

  def _move_order_heuristic(self, board: Board, move: Move, best_move: Optional[Move], depth: int) -> tuple[float, int]:
    """
    | Assigns a heuristic value to moves for ordering.
    | Higher values indicate better moves.
    """
    if move == best_move:
      return (float('inf'), 1) # Prioritize PV move.
    if move in self._killer_moves.get(depth, []):
      return (float('inf'), 0) # Prioritize killer moves.
    return (self._history_heuristic.get(move, self._evaluate(board, move)), 0) # Fallback to history heuristic or explicit board evaluation.

  def _store_killer_move(self, depth: int, move: Move) -> None:
    """
    Stores killer moves for a given depth.
    """
    if depth not in self._killer_moves:
      self._killer_moves[depth] = []
    if move not in self._killer_moves[depth]:
      if len(self._killer_moves[depth]) >= 3:
        self._killer_moves[depth].pop(0)
      self._killer_moves[depth].append(move)

  def _update_history_heuristic(self, move: Move, depth: int) -> None:
    """
    Updates history heuristic table with a move's depth.

    :param move: Move.
    :type move: Move
    :param depth: Depth.
    :type depth: int
    """
    self._history_heuristic[move] = self._history_heuristic.get(move, 0) + 2 ** depth

  def _evaluate(self, board: Board, move: Optional[Move]) -> float:
    """
    Evaluates the given node.

    :param node: Playing board.
    :type node: Board
    :return: Node value.
    :rtype: float
    """
    score = 0
    if move:
      board.play_parsed(move)
    if (board.state is GameState.DRAW or board.state is GameState.NOT_STARTED):
      score = 0
    elif board.current_player_has_won:
      score = float('inf')
    elif board.current_opponent_has_won:
      score = float('-inf')
    else:
      node_hash = board.hash()
      score = self._cached_scores[node_hash]
      if score is None:
        player = board.current_player_color
        opponent = player.opposite
        # Maximize the neighbors of the opponent's queen and minimize our own queen neighbors.
        score = 10 * (board.queen_neighbors_by_color(opponent) - board.queen_neighbors_by_color(player))
        # Maximize our own pieces in play and minimize the opponent's.
        score += 2 * (board.pieces_in_play(player) - board.pieces_in_play(opponent))
        # valid_moves_maximize = node.calculate_valid_moves(current_player, True)
        # collision_count_max = node.count_moves_near_queen(current_player)
        # valid_moves_minimize = node.calculate_valid_moves(current_opponent)
        # collision_count_min = node.count_moves_near_queen(current_opponent)
        # score = (node.count_queen_neighbors(current_opponent) - node.count_queen_neighbors(current_player)) * 20 + (collision_count_max - collision_count_min) * 20 + (len(valid_moves_maximize) - len(valid_moves_minimize)) // 2
        self._cached_scores[node_hash] = score
    if move:
      board.undo()
    return score
