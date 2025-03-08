import re
from typing import Final, Optional
from core.enums import GameType, GameState, PlayerColor, BugType, Direction
from core.game import Position, Bug, Move
from core.hash import ZobristHash

class Board:
  """
  Game Board.
  """

  ORIGIN: Final[Position] = Position(0, 0)
  """
  Position of the first piece played.
  """
  NEIGHBOR_DELTAS: Final[tuple[Position, Position, Position, Position, Position, Position, Position, Position]] = (
    Position(1, 0), # Right
    Position(1, -1), # Up right
    Position(0, -1), # Up left
    Position(-1, 0), # Left
    Position(-1, 1), # Down left
    Position(0, 1), # Down right
    Position(0, 0), # Below (no change)
    Position(0, 0), # Above (no change)
  )
  """
  Offsets of every neighboring tile in each flat direction.
  """

  def __init__(self, gamestring: str = "") -> None:
    """
    Game Board instantation.

    :param gamestring: GameString, defaults to `""`.
    :type gamestring: str, optional
    """
    game_type, state, turn, moves = self._parse_gamestring(gamestring)
    self.type: Final[GameType] = game_type
    self.state: GameState = state
    self.turn: int = turn
    self.move_strings: list[str] = []
    self.moves: list[Optional[Move]] = []
    self._valid_moves_cache: dict[PlayerColor, Optional[set[Move]]] = {
      PlayerColor.WHITE: None,
      PlayerColor.BLACK: None
    }
    self._pos_to_bug: dict[Position, list[Bug]] = {}
    """
    Map for tile positions on the board and bug pieces placed there (pieces can be stacked).
    """
    self._bug_to_pos: dict[Bug, Optional[Position]] = {}
    """
    Map for bug pieces and their current position.  
    Position is None if the piece has not been played yet.  
    Also serves as a check for valid pieces for the game.
    """
    for color in PlayerColor:
      for expansion in self.type:
        if expansion is GameType.BASE:
          self._bug_to_pos[Bug(color, BugType.QUEEN_BEE)] = None
          # Add ids greater than 0 only for bugs with multiple copies.
          for i in range(1, 3):
            self._bug_to_pos[Bug(color, BugType.SPIDER, i)] = None
            self._bug_to_pos[Bug(color, BugType.BEETLE, i)] = None
            self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, i)] = None
            self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, i)] = None
          self._bug_to_pos[Bug(color, BugType.GRASSHOPPER, 3)] = None
          self._bug_to_pos[Bug(color, BugType.SOLDIER_ANT, 3)] = None
        else:
          self._bug_to_pos[Bug(color, BugType(expansion.name))] = None
    self._bugs: Final[list[Bug]] = list(self._bug_to_pos.keys())
    self._art_pos: set[Position] = set()
    self._hash: ZobristHash = ZobristHash(self.type)
    self._play_initial_moves(moves)

  def __str__(self) -> str:
    return f"{self.type};{self.state};{self.current_player_color}[{self.current_player_turn}]{';' if self.moves else ''}{';'.join(self.move_strings)}"

  def __repr__(self):
    return self.__str__()

  @property
  def current_player_color(self) -> PlayerColor:
    """
    Color of the current player.

    :rtype: PlayerColor
    """
    return list(PlayerColor)[self.turn % len(PlayerColor)]

  @property
  def current_player_turn(self) -> int:
    """
    Turn number of the current player.

    :rtype: int
    """
    return 1 + self.turn // 2

  @property
  def current_player_queen_in_play(self) -> bool:
    """
    Whether the current player's queen bee is in play.

    :rtype: bool
    """
    return bool(self._bug_to_pos[Bug(self.current_player_color, BugType.QUEEN_BEE)])

  @property
  def current_player_has_won(self) -> bool:
    """
    Checks whether the current player has won.

    :param color: Player color.
    :type color: PlayerColor
    :return: Whether the current player has won.
    :rtype: bool
    """
    return (self.state is GameState.WHITE_WINS and self.current_player_color is PlayerColor.BLACK) or (self.state is GameState.BLACK_WINS and self.current_player_color is PlayerColor.WHITE)

  @property
  def current_opponent_has_won(self) -> bool:
    """
    Checks whether the current player's opponent has won.

    :param color: Player color.
    :type color: PlayerColor
    :return: Whether the current player's opponent has won.
    :rtype: bool
    """
    return (self.state is GameState.WHITE_WINS and self.current_player_color is PlayerColor.WHITE) or (self.state is GameState.BLACK_WINS and self.current_player_color is PlayerColor.BLACK)

  @property
  def gameover(self) -> bool:
    """
    Whether the game is over.

    :rtype: bool
    """
    return self.state is GameState.DRAW or self.state is GameState.WHITE_WINS or self.state is GameState.BLACK_WINS

  @property
  def valid_moves(self) -> str:
    """
    Current possible legal moves in a joined list of MoveStrings.

    :rtype: str
    """
    return ";".join([self.stringify_move(move) for move in self.calculate_valid_moves_for_player(self.current_player_color)]) or Move.PASS

  def calculate_valid_moves_for_player(self, color: PlayerColor, force: bool = False) -> set[Move]:
    """
    Calculates the set of valid moves for the current player.

    :return: set of valid moves.
    :rtype: set[Move]
    """
    if not self._valid_moves_cache[color] or force:
      moves: set[Move] = set()
      if self.state is GameState.NOT_STARTED or self.state is GameState.IN_PROGRESS:
        self._update_cut_pos()
        for bug, pos in self._bug_to_pos.items():
          # Iterate over available pieces of the current player
          if bug.color is color:
            # Turn 0 is White player's first turn
            if self.turn == 0:
              if self._can_play_on_first_move(bug):
                # Add the only valid placement for the current bug piece
                moves.add(Move(bug, None, Board.ORIGIN))
            # Turn 0 is Black player's first turn
            elif self.turn == 1:
              if self._can_play_on_first_move(bug):
                # Add all valid placements for the current bug piece (can be placed only around the first White player's first piece)
                moves.update(Move(bug, None, self._get_neighbor(Board.ORIGIN, direction)) for direction in Direction.flat())
            # Bug piece has not been played yet
            elif not pos:
              # Check for hand placement and queen placement related rules.
              if self._can_bug_be_played(bug) and self._check_queen_placement(bug):
                # Add all valid placements for the current bug piece
                moves.update(Move(bug, None, placement) for placement in self._get_valid_placements_for_color(color))
            # A bug piece in play can move only if it's at the top and its queen is in play and has not been moved in the previous player's turn
            elif self.current_player_queen_in_play and self.bugs_from_pos(pos)[-1] == bug and self._was_not_last_moved(bug):
              # Can't move pieces that would break the hive. Pieces stacked upon other can never break the hive by moving
              if len(self.bugs_from_pos(pos)) > 1 or self._can_move_without_breaking_hive(pos):
                match bug.type:
                  case BugType.QUEEN_BEE:
                    moves.update(self._get_sliding_moves(bug, pos, 1))
                  case BugType.SPIDER:
                    moves.update(self._get_sliding_moves(bug, pos, 3))
                  case BugType.BEETLE:
                    moves.update(self._get_beetle_moves(bug, pos))
                  case BugType.GRASSHOPPER:
                    moves.update(self._get_grasshopper_moves(bug, pos))
                  case BugType.SOLDIER_ANT:
                    moves.update(self._get_sliding_moves(bug, pos))
                  case BugType.MOSQUITO:
                    moves.update(self._get_mosquito_moves(bug, pos))
                  case BugType.LADYBUG:
                    moves.update(self._get_ladybug_moves(bug, pos))
                  case BugType.PILLBUG:
                    moves.update(self._get_sliding_moves(bug, pos, 1))
                    moves.update(self._get_pillbug_special_moves(pos))
              else:
                match bug.type:
                  case BugType.MOSQUITO:
                    moves.update(self._get_mosquito_moves(bug, pos, True))
                  case BugType.PILLBUG:
                    moves.update(self._get_pillbug_special_moves(pos))
                  case _:
                    pass
      self._valid_moves_cache[color] = moves
    return self._valid_moves_cache[color] or set()

  def play(self, move_string: str):
    """
    Plays the given move.

    :param move_string: MoveString of the move to play.
    :type move_string: str
    :raises ValueError: If the game is over.
    """
    move = self._parse_move(move_string)
    if self.state is GameState.NOT_STARTED:
      self.state = GameState.IN_PROGRESS
    if self.state is GameState.IN_PROGRESS:
      self.turn += 1
      self.move_strings.append(move_string)
      self.moves.append(move)
      self._valid_moves_cache[self.current_player_color] = None
      if move:
        self._bug_to_pos[move.bug] = move.destination
        if move.origin:
          self._pos_to_bug[move.origin].pop()
        if move.destination in self._pos_to_bug:
          self._pos_to_bug[move.destination].append(move.bug)
        else:
          self._pos_to_bug[move.destination] = [move.bug]
        black_queen_surrounded = self.count_queen_neighbors(PlayerColor.BLACK) == 6
        white_queen_surrounded = self.count_queen_neighbors(PlayerColor.WHITE) == 6
        if black_queen_surrounded and white_queen_surrounded:
          self.state = GameState.DRAW
        elif black_queen_surrounded:
          self.state = GameState.WHITE_WINS
        elif white_queen_surrounded:
          self.state = GameState.BLACK_WINS
      self._update_hash()
      return self
    raise ValueError(f"You can't {"play" if move else Move.PASS} when the game is over")

  def undo(self, amount: int = 1) -> None:
    """
    Undoes the specified amount of moves.

    :param amount: Amount of moves to undo, defaults to `1`.
    :type amount: int, optional
    :raises ValueError: If there are not enough moves to undo.
    :raises ValueError: If the game has yet to begin.
    """
    if self.state is not GameState.NOT_STARTED:
      if len(self.moves) >= amount:
        if self.state is not GameState.IN_PROGRESS:
          self.state = GameState.IN_PROGRESS
        for _ in range(amount):
          self.turn -= 1
          self._update_hash()
          self._valid_moves_cache[self.current_player_color] = None
          self.move_strings.pop()
          move = self.moves.pop()
          if move:
            self._pos_to_bug[move.destination].pop()
            self._bug_to_pos[move.bug] = move.origin
            if move.origin:
              self._pos_to_bug[move.origin].append(move.bug)
        if self.turn == 0:
          self.state = GameState.NOT_STARTED
      else:
        raise ValueError(f"Not enough moves to undo: asked for {amount} but only {len(self.moves)} were made")
    else:
      raise ValueError("The game has yet to begin")

  def stringify_move(self, move: Optional[Move]) -> str:
    """
    Returns a MoveString from the given move.

    :param move: Move.
    :type move: Optional[Move]
    :return: MoveString.
    :rtype: str
    """
    if move:
      moved: Bug = move.bug
      relative: Optional[Bug] = None
      direction: Optional[Direction] = None
      if (dest_bugs := self.bugs_from_pos(move.destination)):
        relative = dest_bugs[-1]
      else:
        for neighbor_dir in Direction.flat():
          if (neighbor_bugs := self.bugs_from_pos(self._get_neighbor(move.destination, neighbor_dir))) and (neighbor_bug := neighbor_bugs[0]) != moved:
            relative = neighbor_bug
            direction = neighbor_dir.opposite
            break
      return Move.stringify(moved, relative, direction)
    return Move.PASS

  def count_moves_near_queen(self, color: PlayerColor) -> int:
    """
    Returns the number of available moves that reach the neighboring tiles of the enemy queen bee.

    :param color: Player's color.
    :type color: PlayerColor
    :return: Number of moves that reach the enemy queen bee.
    :rtype: int
    """
    valid_moves = self.calculate_valid_moves_for_player(color, True)

    collision_count = 0
    for move in valid_moves:
      dest = move.destination
      # Get the neighbouring tiles of the destination
      neighbours = [self._get_neighbor(dest, direction) for direction in Direction]
      # Check if the enemy queen bee is in any of the neighbouring tiles
      for pos in neighbours:
        if pos in self._pos_to_bug:
          for bug in self._pos_to_bug[pos]:
            if bug.color.opposite is color and bug.type is BugType.QUEEN_BEE:
              collision_count += 1

    return collision_count

  def count_queen_neighbors(self, color: PlayerColor) -> float:
    """
    Checks whether the specified player's queen is surrounded.

    :param color: Player's color.
    :type color: PlayerColor
    :return: Whether the specified player's queen is surrounded.
    :rtype: bool
    """
    return sum(bool(self.bugs_from_pos(self._get_neighbor(queen_pos, direction))) for direction in Direction.flat()) if (queen_pos := self._bug_to_pos[Bug(color, BugType.QUEEN_BEE)]) else 0

  def bugs_from_pos(self, position: Position) -> list[Bug]:
    """
    Retrieves the list of bug pieces from the given position.

    :param position: Tile position.
    :type position: Position
    :return: The list of bug pieces at the given position.
    :rtype: list[Bug]
    """
    return self._pos_to_bug[position] if position in self._pos_to_bug else []

  def pos_from_bug(self, bug: Optional[Bug]) -> Optional[Position]:
    """
    Retrieves the position of the given bug piece.

    :param bug: Bug piece to get the position of.
    :type bug: Bug
    :return: Position of the given bug piece.
    :rtype: Optional[Position]
    """
    return self._bug_to_pos[bug] if bug and bug in self._bug_to_pos else None

  def hash(self) -> int:
    """
    Returns the current Zobrist Hash value.

    :return: Zobrist Hash value.
    :rtype: int
    """
    return self._hash.value

  def _parse_turn(self, turn: str) -> int:
    """
    Parses a TurnString.

    :param turn: TurnString.
    :type turn: str
    :raises ValueError: If it's not a valid TurnString.
    :return: Turn number.
    :rtype: int
    """
    if (match := re.fullmatch(f"({PlayerColor.WHITE}|{PlayerColor.BLACK})\\[(\\d+)\\]", turn)):
      color, player_turn = match.groups()
      if (turn_number := int(player_turn)) > 0:
        return 2 * turn_number - 2 + list(PlayerColor).index(PlayerColor(color))
      raise ValueError("The turn number must be greater than 0")
    raise ValueError(f"'{turn}' is not a valid TurnString")

  def _parse_gamestring(self, gamestring: str) -> tuple[GameType, GameState, int, list[str]]:
    """
    Parses a GameString.

    :param gamestring: GameString.
    :type gamestring: str
    :raises TypeError: If it's not a valid GameString.
    :return: Tuple of GameString components, namely GameType, GameState, turn number, and list of moves made so far.
    :rtype: tuple[GameType, GameState, int, list[str]]
    """
    values = gamestring.split(";") if gamestring else ["", "", f"{PlayerColor.WHITE}[1]"]
    if len(values) == 1:
      values += ["", f"{PlayerColor.WHITE}[1]"]
    elif len(values) < 3:
      raise TypeError(f"'{gamestring}' is not a valid GameString")
    game_type, state, turn, *moves = values
    return GameType.parse(game_type), GameState.parse(state), self._parse_turn(turn), moves

  def _update_cut_pos(self) -> None:
    if (graph := {pos for pos, bugs in self._pos_to_bug.items() if bugs}):
      new_art_pos: set[Position] = set()
      discovery_times: dict[Position, int] = {}
      low_link_values: dict[Position, int] = {}
      parents: dict[Position, Optional[Position]] = {}
      time: list[int] = [0] # Using list for mutability.
      # Define DFS for Tarjan's algorithm.
      def dfs(u: Position):
        discovery_times[u] = low_link_values[u] = time[0]
        time[0] += 1
        children = 0
        for v in [n for d in Direction.flat() if (n := self._get_neighbor(u, d)) in graph]:
          if v not in discovery_times:
            parents[v] = u
            children += 1
            dfs(v)
            low_link_values[u] = min(low_link_values[u], low_link_values[v])
            if parents.get(u) is None and children > 1:
              new_art_pos.add(u)
            if parents.get(u) is not None and low_link_values[v] >= discovery_times[u]:
              new_art_pos.add(u)
          elif v != parents.get(u):
            low_link_values[u] = min(low_link_values[u], discovery_times[v])
      # Run DFS starting from any node, since the graph is connected.
      dfs(next(iter(graph)))
      # Update current articulation positions.
      self._art_pos.clear()
      self._art_pos.update(new_art_pos)

  def _play_initial_moves(self, moves: list[str]) -> None:
    """
    Make initial moves.

    :param moves: List of MoveStrings.
    :type moves: list[str]
    :raises ValueError: If the amount of moves to make is not coherent with the turn number.
    """
    if self.turn == len(moves):
      old_turn = self.turn
      old_state = self.state
      self.turn = 0
      self.state = GameState.NOT_STARTED
      for move in moves:
        self.play(move)
      if old_turn != self.turn:
        raise ValueError(f"TurnString is not correct, should be {self.current_player_color}[{self.current_player_turn}]")
      if old_state != self.state:
        raise ValueError(f"GameStateString is not correct, should be {self.state}")
    else:
      raise ValueError(f"Expected {self.turn} moves but got {len(moves)}")

  def _get_valid_placements_for_color(self, color: PlayerColor) -> set[Position]:
    """
    Calculates all valid placements for the current player.

    :return: Set of valid positions where new pieces can be placed.
    :rtype: set[Position]
    """
    placements: set[Position] = set()
    # Iterate over all placed bug pieces of the current player
    for bug, pos in self._bug_to_pos.items():
      if bug.color is color and pos and self._is_bug_on_top(bug):
        # Iterate over all neighbors of the current bug piece
        for direction in Direction.flat():
          neighbor = self._get_neighbor(pos, direction)
          # If the neighboring tile is empty
          if not self.bugs_from_pos(neighbor):
            # If all neighbor's neighbors are empty or of the same color, add the neighbor as a valid placement
            if all(not self.bugs_from_pos(self._get_neighbor(neighbor, dir)) or self.bugs_from_pos(self._get_neighbor(neighbor, dir))[-1].color is color for dir in Direction.flat() if dir is not direction.opposite):
              placements.add(neighbor)
    return placements

  def _get_sliding_moves(self, bug: Bug, origin: Position, depth: int = 0) -> set[Move]:
    """
    Calculates the set of valid sliding moves, optionally with a fixed depth.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param depth: Optional fixed depth of the move, defaults to `0`.
    :type depth: int, optional
    :return: Set of valid sliding moves.
    :rtype: set[Move]
    """
    destinations: set[Position] = set()
    visited: set[Position] = set()
    stack: set[tuple[Position, int]] = {(origin, 0)}
    unlimited_depth = depth == 0
    while stack:
      current, current_depth = stack.pop()
      visited.add(current)
      if unlimited_depth or current_depth == depth:
        destinations.add(current)
      if unlimited_depth or current_depth < depth:
        stack.update((neighbor, current_depth + 1) for direction in Direction.flat() if (neighbor := self._get_neighbor(current, direction)) not in visited and not self.bugs_from_pos(neighbor) and self._check_for_door(origin, current, direction))
    return {Move(bug, origin, destination) for destination in destinations if destination != origin}

  def _check_for_door(self, origin: Position, position: Position, direction: Direction) -> bool:
    """
    Checks whether a bug piece can slide from origin to position (no door formation).

    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param position: Destination position.
    :type position: Position
    :param direction: Moving direction.
    :type direction: Direction
    :return: Whether a bug piece can slide from origin to position.
    :rtype: bool
    """
    return bool(self.bugs_from_pos((right := self._get_neighbor(position, direction.right_of)))) != bool(self.bugs_from_pos((left := self._get_neighbor(position, direction.left_of)))) and right != origin != left

  def _get_beetle_moves(self, bug: Bug, origin: Position, virtual: bool = False) -> set[Move]:
    """
    Calculates the set of valid moves for a Beetle.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param virtual: Whether the bug is not at origin, and is just passing by as part of its full move, defaults to `False`.
    :type virtual: bool, optional
    :return: Set of valid Beetle moves.
    :rtype: set[Move]
    """
    moves: set[Move] = set()
    for direction in Direction.flat():
      # Don't consider the Beetle in the height, unless it's a virtual move (the bug is not actually in origin, but moving at the top of origin is part of its full move).
      height = len(self.bugs_from_pos(origin)) - 1 + virtual
      destination = self._get_neighbor(origin, direction)
      dest_height = len(self.bugs_from_pos(destination))
      left_height = len(self.bugs_from_pos(self._get_neighbor(origin, direction.left_of)))
      right_height = len(self.bugs_from_pos(self._get_neighbor(origin, direction.right_of)))
      # Logic from http://boardgamegeek.com/wiki/page/Hive_FAQ#toc9
      if not ((height == 0 and dest_height == 0 and left_height == 0 and right_height == 0) or (dest_height < left_height and dest_height < right_height and height < left_height and height < right_height)):
        moves.add(Move(bug, origin, destination))
    return moves

  def _get_grasshopper_moves(self, bug: Bug, origin: Position) -> set[Move]:
    """
    Calculates the set of valid moves for a Grasshopper.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Grasshopper moves.
    :rtype: set[Move]
    """
    moves: set[Move] = set()
    for direction in Direction.flat():
      destination: Position = self._get_neighbor(origin, direction)
      distance: int = 0
      while self.bugs_from_pos(destination):
        # Jump one more tile in the same direction
        destination = self._get_neighbor(destination, direction)
        distance += 1
      if distance > 0:
        # Can only move if there's at least one piece in the way
        moves.add(Move(bug, origin, destination))
    return moves

  def _get_mosquito_moves(self, bug: Bug, origin: Position, special_only: bool = False) -> set[Move]:
    """
    Calculates the set of valid Mosquito moves, which copies neighboring bug pieces moves, and can be either normal or special (Pillbug) moves depending on the special_only flag.

    :param bug: Mosquito bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :param special_only: Whether to include special moves only, defaults to `False`.
    :type special_only: bool, optional
    :return: Set of valid Mosquito moves.
    :rtype: set[Move]
    """
    if len(self.bugs_from_pos(origin)) > 1:
      return self._get_beetle_moves(bug, origin)
    moves: set[Move] = set()
    bugs_copied: set[BugType] = set()
    for direction in Direction.flat():
      if (bugs := self.bugs_from_pos(self._get_neighbor(origin, direction))) and (neighbor := bugs[-1]).type not in bugs_copied:
        bugs_copied.add(neighbor.type)
        if special_only:
          if neighbor.type == BugType.PILLBUG:
            moves.update(self._get_pillbug_special_moves(origin))
        else:
          match neighbor.type:
            case BugType.QUEEN_BEE:
              moves.update(self._get_sliding_moves(bug, origin, 1))
            case BugType.SPIDER:
              moves.update(self._get_sliding_moves(bug, origin, 3))
            case BugType.BEETLE:
              moves.update(self._get_beetle_moves(bug, origin))
            case BugType.GRASSHOPPER:
              moves.update(self._get_grasshopper_moves(bug, origin))
            case BugType.SOLDIER_ANT:
              moves.update(self._get_sliding_moves(bug, origin))
            case BugType.LADYBUG:
              moves.update(self._get_ladybug_moves(bug, origin))
            case BugType.PILLBUG:
              moves.update(self._get_sliding_moves(bug, origin, 1))
            case BugType.MOSQUITO:
              pass
    return moves

  def _get_ladybug_moves(self, bug: Bug, origin: Position) -> set[Move]:
    """
    Calculates the set of valid moves for a Ladybug.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Ladybug moves.
    :rtype: set[Move]
    """
    return {
      Move(bug, origin, final_move.destination)
      for first_move in self._get_beetle_moves(bug, origin, True) if self.bugs_from_pos(first_move.destination)
      for second_move in self._get_beetle_moves(bug, first_move.destination, True) if self.bugs_from_pos(second_move.destination) and second_move.destination != origin
      for final_move in self._get_beetle_moves(bug, second_move.destination, True) if not self.bugs_from_pos(final_move.destination) and final_move.destination != origin
    }

  def _get_pillbug_special_moves(self, origin: Position) -> set[Move]:
    """
    Calculates the set of valid special Pillbug moves.

    :param origin: Position of the Pillbug.
    :type origin: Position
    :return: Set of valid special Pillbug moves.
    :rtype: set[Move]
    """
    moves: set[Move] = set()
    # There must be at least one empty neighboring tile for the Pillbug to move another bug piece
    if (empty_positions := [self._get_neighbor(origin, direction) for direction in Direction.flat() if not self.bugs_from_pos(self._get_neighbor(origin, direction))]):
      for direction in Direction.flat():
        position = self._get_neighbor(origin, direction)
        # A Pillbug can move another bug piece only if it's not stacked, it's not the last moved piece, it can be moved without breaking the hive, and it's not obstructed in moving above the Pillbug itself
        if len(bugs := self.bugs_from_pos(position)) == 1 and self._was_not_last_moved(neighbor := bugs[-1]) and self._can_move_without_breaking_hive(position) and Move(neighbor, position, origin) in self._get_beetle_moves(neighbor, position):
          moves.update(Move(neighbor, position, move.destination) for move in self._get_beetle_moves(neighbor, position, True) if move.destination in empty_positions)
    return moves

  def _can_move_without_breaking_hive(self, position: Position) -> bool:
    """
    Checks whether a bug piece can be moved from the given position.

    :param position: Position where the bug piece is located.
    :type position: Position
    :return: Whether a bug piece in the given position can move.
    :rtype: bool
    """
    return position in self._art_pos

  def _can_play_on_first_move(self, bug: Bug) -> bool:
    """
    Checks whether the given bug piece can be played on the current player's first move.

    :param bug: Bug piece.
    :type bug: Bug
    :return: Whether the bug piece can be played.
    :rtype: bool
    """
    return bug.type is not BugType.QUEEN_BEE and self._can_bug_be_played(bug)

  def _check_queen_placement(self, bug: Bug) -> bool:
    """
    Checks for queen placement related rule.

    :param bug: Bug piece to check.
    :type bug: Bug
    :return: Whether the bug piece can be placed.
    :rtype: bool
    """
    return self.current_player_turn != 4 or self.current_player_queen_in_play or bug.type is BugType.QUEEN_BEE

  def _can_bug_be_played(self, piece: Bug) -> bool:
    """
    Checks whether the given bug piece can be drawn from hand (has the lowest ID among same-type pieces).

    :param piece: Bug piece to check.
    :type piece: Bug
    :return: Whether the given bug piece can be played.
    :rtype: bool
    """
    return all(bug.id >= piece.id for bug, pos in self._bug_to_pos.items() if pos is None and bug.type is piece.type and bug.color is piece.color)

  def _was_not_last_moved(self, bug: Bug) -> bool:
    """
    Checks whether the given bug piece was not moved in the previous turn.

    :param bug: Bug piece.
    :type bug: Bug
    :return: Whether the bug piece was not last moved.
    :rtype: bool
    """
    return not self.moves[-1] or self.moves[-1].bug != bug

  def _parse_move(self, move_string: str) -> Optional[Move]:
    """
    Parses a MoveString.

    :param move_string: MoveString.
    :type move_string: str
    :raises ValueError: If move_string is 'pass' but there are other valid moves.
    :raises ValueError: If move_string is not a valid move for the current board state.
    :raises ValueError: If bug_string_2 has not been played yet.
    :raises ValueError: If more than one direction was specified.
    :raises ValueError: If move_string is not a valid MoveString.
    :return: Move.
    :rtype: Optional[Move]
    """
    if move_string == Move.PASS:
      if not self.calculate_valid_moves_for_player(self.current_player_color):
        return None
      raise ValueError("You can't pass when you have valid moves")
    if (match := re.fullmatch(Move.REGEX, move_string)):
      bug_string_1, _, _, _, _, left_dir, bug_string_2, _, _, _, right_dir = match.groups()
      if not left_dir or not right_dir:
        moved = Bug.parse(bug_string_1)
        if (relative_pos := self.pos_from_bug(Bug.parse(bug_string_2)) if bug_string_2 else Board.ORIGIN):
          move = Move(moved, self.pos_from_bug(moved), self._get_neighbor(relative_pos, Direction(f"{left_dir}|") if left_dir else Direction(f"|{right_dir or ""}")))
          if move in self.calculate_valid_moves_for_player(self.current_player_color):
            return move
          raise ValueError(f"'{move_string}' is not a valid move for the current board state")
        raise ValueError(f"'{bug_string_2}' has not been played yet")
      raise ValueError("Only one direction at a time can be specified")
    raise ValueError(f"'{move_string}' is not a valid MoveString")

  def _is_bug_on_top(self, bug: Bug) -> bool:
    """
    Checks if the given bug has been played and is at the top of the stack.

    :param bug: Bug piece.
    :type bug: Bug
    :return: Whether the bug is at the top.
    :rtype: bool
    """
    return (pos := self.pos_from_bug(bug)) is not None and self.bugs_from_pos(pos)[-1] == bug

  def _get_neighbor(self, position: Position, direction: Direction) -> Position:
    """
    Returns the neighboring position from the given direction.

    :param position: Central position.
    :type position: Position
    :param direction: Direction of movement.
    :type direction: Direction
    :return: Neighboring position in the specified direction.
    :rtype: Position
    """
    return position + Board.NEIGHBOR_DELTAS[direction.delta_index]

  def _update_hash(self) -> None:
    self._hash.toggle_turn()
    if len(self.moves) > 1 and (second_last_move := self.moves[-2]) is not None:
      self._hash.toggle_last_moved_piece(self._bugs.index(second_last_move.bug))
    if len(self.moves) > 0 and (last_move := self.moves[-1]) is not None:
      self._hash.toggle_last_moved_piece(self._bugs.index(last_move.bug))
      if (origin := last_move.origin) is not None:
        self._hash.toggle_piece(self._bugs.index(last_move.bug), origin, len(self.bugs_from_pos(origin)))
      self._hash.toggle_piece(self._bugs.index(last_move.bug), last_move.destination, len(self.bugs_from_pos(last_move.destination)) - 1)
