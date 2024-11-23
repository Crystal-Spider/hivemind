from typing import Final, Set
from enums import GameType, GameState, PlayerColor, BugType, Direction
from game import Position, Bug, Move
import re

class Board():
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

    :param gamestring: GameString, defaults to "".
    :type gamestring: str, optional
    """
    type, state, turn, moves = self._parse_gamestring(gamestring)
    self.type: Final[GameType] = type
    self.state: GameState = state
    self.turn: int = turn
    self.moves: list[Move] = []
    self.move_strings: list[str] = moves # TODO: Implement initial moves.
    self._pos_to_bug: dict[Position, list[Bug]] = {}
    """
    Map for tile positions on the board and bug pieces placed there (pieces can be stacked).
    """
    self._bug_to_pos: dict[Bug, Position | None] = {}
    """
    Map for bug pieces and their current position.  
    Position is None if the piece has not been played yet.  
    Also serves as a check for valid pieces for the game.
    """
    for color in PlayerColor:
      for expansion in self.type:
        if expansion is GameType.Base:
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
    self._valid_moves_cache: Set[Move] | None = None
    # TODO: Most likely, a cache for the best move will be needed too.
    # TODO: Implement support for Move.PASS (play, undo, stringify, etc.)
    # TODO: Check for gameovers (play, undo).

  def __str__(self) -> str:
    return f"{self.type};{self.state};{self.current_player_color}[{self.current_player_turn}]{';' if len(self.moves) else ''}{';'.join(self.move_strings)}"

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
      return 2 * int(player_turn) - 2 + list(PlayerColor).index(PlayerColor(color))
    else:
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
    type, state, turn, *moves = values
    return GameType.parse(type), GameState.parse(state), self._parse_turn(turn), moves

  def _make_initial_moves(self, moves: list[str]) -> list[str]:
    if self.turn - 1 == len(moves):
      for move in moves:
        self.play(move)
      return moves
    raise ValueError(f"Expected {self.turn - 1} moves but got {len(moves)}")

  def play(self, move_string: str) -> None:
    """
    Plays the given move.

    :param move_string: MoveString of the move to play.
    :type move_string: str
    :raises ValueError: If the game is over.
    """
    move = self.parse_move(move_string)
    if self.state is GameState.NOT_STARTED:
      self.state = GameState.IN_PROGRESS
    if self.state is GameState.IN_PROGRESS:
      self.turn += 1
      self.moves.append(move)
      self.move_strings.append(move_string)
      self._valid_moves_cache = None
      self._bug_to_pos[move.bug] = move.destination
      if move.origin:
        self._pos_to_bug[move.origin].pop()
      if move.destination in self._pos_to_bug:
        self._pos_to_bug[move.destination].append(move.bug)
      else:
        self._pos_to_bug[move.destination] = [move.bug]
    else:
      raise ValueError("You can't play, the game is over")

  def undo(self, amount: int = 1) -> None:
    if self.state is not GameState.NOT_STARTED:
      if len(self.moves) >= amount:
        if self.state is not GameState.IN_PROGRESS:
          self.state = GameState.IN_PROGRESS
        for _ in range(amount):
          self.turn -= 1
          self.move_strings.pop()
          move = self.moves.pop()
          self._valid_moves_cache = None
          self._pos_to_bug[move.destination].pop()
          self._bug_to_pos[move.bug] = move.origin
          if move.origin:
            self._pos_to_bug[move.origin].append(move.bug)
        if self.turn == 0:
          self.state = GameState.NOT_STARTED
      else:
        raise ValueError(f"Not enough moves to undo: asked for {amount} but only {len(self.moves)} were made")
    else:
      raise ValueError(f"The game has yet to begin")

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
  def valid_moves(self) -> str:
    """
    Current possible legal moves in a joined list of MoveStrings.

    :rtype: str
    """
    return ";".join([self.stringify_move(move) for move in self.get_valid_moves()]) or Move.PASS

  @property
  def best_move(self) -> str:
    """
    Current best move as a MoveString.

    :rtype: str
    """
    move = self.get_best_move()
    return self.stringify_move(move) if move else Move.PASS
  
  def stringify_move(self, move: Move) -> str:
    """
    Returns a MoveString from the given move.

    :param move: Move.
    :type move: Move
    :return: MoveString.
    :rtype: str
    """
    moved: Bug = move.bug
    relative: Bug | None = None
    direction: Direction | None = None
    if (dest_bugs := self.bugs_from_pos(move.destination)):
      relative = dest_bugs[-1]
    else:
      for dir in Direction.flat():
        if (neighbor_bugs := self.bugs_from_pos(self.get_neighbor(move.destination, dir))):
          relative = neighbor_bugs[0]
          direction = dir.opposite
          break
    return Move.stringify(moved, relative, direction)
  
  def parse_move(self, move_string: str) -> Move:
    """
    Parses a MoveString.

    :param move_string: MoveString.
    :type move_string: str
    :raises ValueError: If move_string is not a valid move for the current board state.
    :raises ValueError: If bug_string_2 has not been played yet.
    :raises ValueError: If more than one direction was specified.
    :raises ValueError: If move_string is not a valid MoveString.
    :return: Move.
    :rtype: Move
    """
    if (match := re.fullmatch(Move.REGEX, move_string)):
      bug_string_1, _, _, _, _, left_dir, bug_string_2, _, _, _, right_dir = match.groups()
      if not left_dir or not right_dir:
        moved = Bug.parse(bug_string_1)
        if (relative_pos := self.pos_from_bug(Bug.parse(bug_string_2)) if bug_string_2 else self.ORIGIN):
          move = Move(moved, self.pos_from_bug(moved), self.get_neighbor(relative_pos, Direction(f"{left_dir}|") if left_dir else Direction(f"|{right_dir or ""}")))
          if move in self.get_valid_moves():
            return move
          raise ValueError(f"'{move_string}' is not a valid move for the current board state")
        raise ValueError(f"'{bug_string_2}' has not been played yet")
      raise ValueError(f"Only one direction at a time can be specified")
    raise ValueError(f"'{move_string}' is not a valid MoveString")

  def bugs_from_pos(self, position: Position) -> list[Bug]:
    """
    Retrieves the list of bug pieces from the given position.

    :param position: Tile position.
    :type position: Position
    :return: The list of bug pieces at the given position.
    :rtype: list[Bug]
    """
    return self._pos_to_bug[position] if position in self._pos_to_bug else []

  def pos_from_bug(self, bug: Bug) -> Position | None:
    """
    Retrieves the position of the given bug piece.

    :param bug: Bug piece to get the position of.
    :type bug: Bug
    :return: Position of the given bug piece.
    :rtype: Position | None
    """
    return self._bug_to_pos[bug] if bug in self._bug_to_pos else None

  def get_valid_moves(self) -> Set[Move]:
    if not self._valid_moves_cache:
      self._valid_moves_cache = set()
      if self.state is GameState.NOT_STARTED or self.state is GameState.IN_PROGRESS:
        for bug, pos in self._bug_to_pos.items():
          # Iterate over available pieces of the current player
          if bug.color is self.current_player_color:
            # Turn 0 is White player's first turn
            if self.turn == 0:
              # Can't place the queen on the first turn
              if bug.type is not BugType.QUEEN_BEE and self.can_bug_be_played(bug):
                # Add the only valid placement for the current bug piece
                self._valid_moves_cache.add(Move(bug, None, self.ORIGIN))
            # Turn 0 is Black player's first turn
            elif self.turn == 1:
              # Can't place the queen on the first turn
              if bug.type is not BugType.QUEEN_BEE and self.can_bug_be_played(bug):
                # Add all valid placements for the current bug piece (can be placed only around the first White player's first piece)
                self._valid_moves_cache.update([Move(bug, None, self.get_neighbor(self.ORIGIN, direction)) for direction in Direction.flat()])
            # Bug piece has not been played yet
            elif not pos:
              # Check hand placement, and turn and queen placement, related rule.
              if self.can_bug_be_played(bug) and (self.current_player_turn != 4 or (self.current_player_turn == 4 and (self.current_player_queen_in_play or (not self.current_player_queen_in_play and bug.type is BugType.QUEEN_BEE)))):
                # Add all valid placements for the current bug piece
                self._valid_moves_cache.update([Move(bug, None, placement) for placement in self.get_valid_placements()])
            # A bug piece in play can move only if it's at the top and its queen is in play 
            elif self.current_player_queen_in_play and self.bugs_from_pos(pos)[-1] == bug:
              # Can't move pieces that would break the hive. Pieces stacked upon other can never break the hive by moving.
              if self.bugs_from_pos(pos) or self.can_move_without_breaking_hive(pos):
                match bug.type:
                  case BugType.QUEEN_BEE:
                    self._valid_moves_cache.update(self.get_sliding_moves(bug, pos, 1))
                  case BugType.SPIDER:
                    self._valid_moves_cache.update(self.get_sliding_moves(bug, pos, 3))
                  case BugType.BEETLE:
                    self._valid_moves_cache.update(self.get_beetle_moves(bug, pos))
                  case BugType.GRASSHOPPER:
                    self._valid_moves_cache.update(self.get_grasshopper_moves(bug, pos))
                  case BugType.SOLDIER_ANT:
                    self._valid_moves_cache.update(self.get_sliding_moves(bug, pos))
                  case BugType.MOSQUITO:
                    pass
                  case BugType.LADYBUG:
                    self._valid_moves_cache.update(self.get_ladybug_moves(bug, pos))
                  case BugType.PILLBUG:
                    self._valid_moves_cache.update(self.get_sliding_moves(bug, pos, 1))
              else:
                match bug.type:
                  case BugType.MOSQUITO:
                    pass
                  case BugType.PILLBUG:
                    pass
                  case _:
                    pass
    return self._valid_moves_cache
  
  def get_best_move(self) -> Move | None:
    """
    Get the best move to play.  
    Currently, simply returns the first valid move.

    :return: The best possible move to play.
    :rtype: Move
    """
    return list(self.get_valid_moves())[0] if self.get_valid_moves() else None

  def can_bug_be_played(self, piece: Bug) -> bool:
    """
    Checks whether the given bug piece can be drawn from hand (has the lowest ID among same-type pieces).

    :param piece: Bug piece to check.
    :type piece: Bug
    :return: Whether the given bug piece can be played.
    :rtype: bool
    """
    return all(bug.id >= piece.id for bug, pos in self._bug_to_pos.items() if pos is None and bug.type is piece.type)

  def is_bug_on_top(self, bug: Bug) -> bool:
    """
    Checks if the given bug has been played and is at the top of the stack.

    :param bug: Bug piece.
    :type bug: Bug
    :return: Whether the bug is at the top.
    :rtype: bool
    """
    return (pos := self.pos_from_bug(bug)) != None and self.bugs_from_pos(pos)[-1] == bug

  def get_valid_placements(self) -> Set[Position]:
    """
    Calculates all valid placements for the current player.

    :return: Set of valid positions where new pieces can be placed.
    :rtype: Set[Position]
    """
    placements: Set[Position] = set()
    # Iterate over all placed bug pieces of the current player
    for bug, pos in self._bug_to_pos.items():
      if bug.color is self.current_player_color and pos and self.is_bug_on_top(bug):
        # Iterate over all neighbors of the current bug piece
        for direction in Direction.flat():
          neighbor = self.get_neighbor(pos, direction)
          # If the neighboring tile is empty
          if not self.bugs_from_pos(neighbor):
            # If all neighbor's neighbors are empty or of the same color, add the neighbor as a valid placement
            if all(not self.bugs_from_pos(self.get_neighbor(neighbor, dir)) or self.bugs_from_pos(self.get_neighbor(neighbor, dir))[-1].color is self.current_player_color for dir in Direction.flat() if dir is not direction.opposite):
              placements.add(neighbor)
    return placements

  def get_neighbor(self, position: Position, direction: Direction) -> Position:
    return position + self.NEIGHBOR_DELTAS[direction.delta_index]

  def can_move_without_breaking_hive(self, position: Position) -> bool:
    # Try gaps heuristic first
    neighbors: list[list[Bug]] = [self.bugs_from_pos(self.get_neighbor(position, direction)) for direction in Direction.flat()]
    # If there is more than 1 gap, perform a DFS to check if all neighbors are still connected in some way.
    if sum(bool(neighbors[i] and not neighbors[i - 1]) for i in range(len(neighbors))) > 1:
      visited: Set[Position] = set()
      neighbors_pos: list[Position] = [pos for bugs in neighbors for pos in [self.pos_from_bug(bugs[-1])] if bugs if pos]    
      stack: Set[Position] = {neighbors_pos[0]}
      while stack:
        current = stack.pop()
        visited.add(current)
        stack.update(neighbor for direction in Direction.flat() if (neighbor := self.get_neighbor(current, direction)) != position and self.bugs_from_pos(neighbor) and neighbor not in visited)
      # Check if all neighbors with bug pieces were visited
      return all(neighbor_pos in visited for neighbor_pos in neighbors_pos)
    # If there is only 1 gap, then all neighboring pieces are connected even without the piece at the given position.
    return True

  def get_beetle_moves(self, bug: Bug, origin: Position) -> Set[Move]:
    """
    Calculates the set of valid moves for a Beetle.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Beetle moves.
    :rtype: Set[Move]
    """
    moves: Set[Move] = set()
    for direction in Direction.flat():
      height = len(self.bugs_from_pos(origin)) - 1 # Don't consider the Beetle
      destination = self.get_neighbor(origin, direction)
      dest_height = len(self.bugs_from_pos(destination))
      left_height = len(self.bugs_from_pos(self.get_neighbor(origin, direction.left_of)))
      right_height = len(self.bugs_from_pos(self.get_neighbor(origin, direction.right_of)))
      # Logic from http://boardgamegeek.com/wiki/page/Hive_FAQ#toc9
      if not ((height == 0 and dest_height == 0 and left_height == 0 and right_height == 0) or (dest_height < left_height and dest_height < right_height and height < left_height and height < right_height)):
        moves.add(Move(bug, origin, destination)) # TODO: Is the height / direction needed?
    return moves

  def get_grasshopper_moves(self, bug: Bug, origin: Position) -> Set[Move]:
    """
    Calculates the set of valid moves for a Grasshopper.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Grasshopper moves.
    :rtype: Set[Move]
    """
    moves: Set[Move] = set()
    for direction in Direction.flat():
      destination: Position = self.get_neighbor(origin, direction)
      distance: int = 0
      while self.bugs_from_pos(destination):
        # Jump one more tile in the same direction
        destination = self.get_neighbor(origin, direction)
        distance += 1
      if distance > 0:
        # Can only move if there's at least one piece in the way
        moves.add(Move(bug, origin, destination))
    return moves

  def get_mosquito_moves(self, bug: Bug, origin: Position):
    pass

  def get_ladybug_moves(self, bug: Bug, origin: Position) -> Set[Move]:
    """
    Calculates the set of valid moves for a Ladybug.

    :param bug: Moving bug piece.
    :type bug: Bug
    :param origin: Initial position of the bug piece.
    :type origin: Position
    :return: Set of valid Ladybug moves.
    :rtype: Set[Move]
    """
    return {
      Move(bug, origin, origin)
      for first_move in self.get_beetle_moves(bug, origin) if len(self.bugs_from_pos(first_move.destination))
      for second_move in self.get_beetle_moves(bug, first_move.destination) if len(self.bugs_from_pos(second_move.destination))
      for final_move in self.get_beetle_moves(bug, second_move.destination) if len(self.bugs_from_pos(final_move.destination)) and final_move.destination != origin
    }

  def get_pillbug_special_moves(self, bug: Bug, origin: Position):
    pass

  def get_sliding_moves(self, bug: Bug, origin: Position, depth: int = 0):
    destinations: Set[Position] = set()
    visited: Set[Position] = set()
    stack: Set[tuple[Position, int]] = {(origin, 0)}
    unlimited_depth = depth == 0
    while stack:
      current, current_depth = stack.pop()
      visited.add(current)
      if unlimited_depth or current_depth == depth:
        destinations.add(current)
      if unlimited_depth or current_depth < depth:
        stack.update(
          (neighbor, current_depth + 1)
          for direction in Direction.flat()
          if (neighbor := self.get_neighbor(current, direction)) not in visited and not self.bugs_from_pos(neighbor) and bool(self.bugs_from_pos(self.get_neighbor(current, direction.right_of))) != bool(self.bugs_from_pos(self.get_neighbor(current, direction.left_of)))
        )
    return {Move(bug, origin, destination) for destination in destinations if destination != origin}