from typing import Optional
from random import choice
from time import sleep, time
from copy import deepcopy
from abc import ABC, abstractmethod
from core.board import Board
from core.game import Move
from core.enums import GameState, PlayerColor
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
    Empties the current cache for the best move.  
    Might empty more data depending on the agent.
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
    self._pv_table: dict[int, str] = {}
    self._killer_moves: dict[int, list[str]] = {}
    self._history_heuristic: dict[str, int] = {}
    self._cached_scores: ScoreTable = ScoreTable()
    self._visited_nodes: int = 0
    self._cutoffs: int = 0

  def _find_best_move(self, board: Board, max_branching_factor: int, max_depth: int = 0, time_limit: int = 0) -> str:
    start_time = time()
    maximizing = board.current_player_color is PlayerColor.WHITE
    best_move = None
    scores: list[tuple[Optional[str], float]] = []
    depth = 0
    while not max_depth or depth < max_depth:
      depth += 1
      best_move, score = self._alpha_beta_search(board, max_branching_factor, depth, float('-inf'), float('inf'), maximizing, start_time, time_limit)
      scores.append((best_move, score))
      if time_limit and time() - start_time > time_limit:
        break
    self._transpos_table.flush()
    print(f"Visited nodes: {self._visited_nodes}; Cutoffs: {self._cutoffs}; Scores: {scores}")
    self._visited_nodes = 0
    self._cutoffs = 0
    return best_move or Move.PASS

  def _alpha_beta_search(self, node: Board, max_branching_factor: int, depth: int, alpha: float, beta: float, maximizing: bool, start_time: float, time_limit: int) -> tuple[Optional[str], float]:
    self._visited_nodes += 1
    node_hash = node.hash()
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

    if depth == 0 or node.gameover:
      return None, self._evaluate(node)

    best_move = self._pv_table.get(node_hash, None)
    children = self._gen_children(node)
    children.sort(key=lambda x: self._move_order_heuristic(x[0], x[1], best_move, depth), reverse=maximizing)
    if len(children) > max_branching_factor:
      del children[max_branching_factor:]

    if maximizing:
      max_value = float('-inf')
      for child, move in children:
        _, value = self._alpha_beta_search(child, max_branching_factor, depth - 1, alpha, beta, not maximizing, start_time, time_limit)
        if max_value < value:
          max_value = value
          best_move = move
        alpha = max(alpha, max_value)
        if alpha >= beta:
          self._cutoffs += 1
          self._store_killer_move(depth, move)
          break # Beta cut-off
      self._transpos_table[node_hash] = TranspositionTableEntry(TranspositionTableEntryType.EXACT if max_value < beta else TranspositionTableEntryType.LOWER_BOUND, max_value, depth, best_move)
      if best_move:
        self._pv_table[node_hash] = best_move
        self._update_history_heuristic(best_move, depth)
      return best_move, max_value
    else:
      min_value = float('inf')
      for child, move in children:
        _, value = self._alpha_beta_search(child, max_branching_factor, depth - 1, alpha, beta, not maximizing, start_time, time_limit)
        if min_value > value:
          min_value = value
          best_move = move
        beta = min(beta, min_value)
        if alpha >= beta:
          self._cutoffs += 1
          self._store_killer_move(depth, move)
          break # Alpha cut-off
      self._transpos_table[node_hash] = TranspositionTableEntry(TranspositionTableEntryType.EXACT if min_value > alpha else TranspositionTableEntryType.UPPER_BOUND, min_value, depth, best_move)
      if best_move:
        self._pv_table[node_hash] = best_move
        self._update_history_heuristic(best_move, depth)
      return best_move, min_value

  def _move_order_heuristic(self, board: Board, move: str, best_move: Optional[str], depth: int) -> tuple[float, int]:
    """
    Assigns a heuristic value to moves for ordering.  
    Higher values indicate better moves.
    """
    if move == best_move:
      return (float('inf'), 1) # Prioritize PV move
    if move in self._killer_moves.get(depth, []):
      return (float('inf'), 0) # Prioritize killer moves
    return (self._history_heuristic.get(move, self._evaluate(board)), 0) # Fallback to history heuristic or explicit board evaluation

  def _store_killer_move(self, depth: int, move: str) -> None:
    """
    Stores killer moves for a given depth.
    """
    if depth not in self._killer_moves:
      self._killer_moves[depth] = []
    if move not in self._killer_moves[depth]:
      if len(self._killer_moves[depth]) >= 3:
        self._killer_moves[depth].pop(0)
      self._killer_moves[depth].append(move)

  def _update_history_heuristic(self, move: str, depth: int) -> None:
    """
    Updates history heuristic table with a move's depth.
    """
    self._history_heuristic[move] = self._history_heuristic.get(move, 0) + 2 ** depth

  def _gen_children(self, parent: Board) -> list[tuple[Board, str]]:
    """
    Generates valid children from the given parent.

    :param parent: Parent node.
    :type parent: Board
    :return: List of children.
    :rtype: list[tuple[Board, str]]
    """
    return [(deepcopy(parent).play(move), move) for move in parent.valid_moves.split(";") if not parent.gameover]

  def _evaluate(self, node: Board) -> float:
    """
    Evaluates the given node.  
    Currently, it's a very naive implementation that weights the winning state (how many pieces surround the enemy queen minus how many pieces surround yours) and the mobility state (amount of your available moves minus the enemy's).

    :param node: Playing board.
    :type node: Board
    :return: Node value.
    :rtype: float
    """

    match node.state:
      case GameState.WHITE_WINS:
        return float('inf')
      case GameState.BLACK_WINS:
        return float('-inf')
      case GameState.DRAW | GameState.NOT_STARTED:
        return 0
      case _: pass

    node_hash = node.hash()
    score = self._cached_scores[node_hash]
    if score is None:
      valid_moves_maximize = node.calculate_valid_moves_for_player(PlayerColor.WHITE, True)
      collision_count_max = node.count_moves_near_queen(PlayerColor.WHITE)
      valid_moves_minimize = node.calculate_valid_moves_for_player(PlayerColor.BLACK)
      collision_count_min = node.count_moves_near_queen(PlayerColor.BLACK)
      score = (node.count_queen_neighbors(PlayerColor.BLACK) - node.count_queen_neighbors(PlayerColor.WHITE)) * 20 + (collision_count_max-collision_count_min) * 20 + (len(valid_moves_maximize) - len(valid_moves_minimize)) // 2
      self._cached_scores[node_hash] = score
    return score
