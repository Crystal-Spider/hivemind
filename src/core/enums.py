from typing import Final
from enum import StrEnum, Flag, auto
from functools import reduce

class Command(StrEnum):
  """
  Available command.
  """
  INFO = "info"
  """
  | Displays the identifier string of the engine and list of its capabilities.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#info.
  """
  HELP = "help"
  """
  | Displays the list of available commands.
  | If a command is specified, displays the help for that command.
  """
  OPTIONS = "options"
  """
  | Displays the available options for the engine.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#options.
  """
  NEWGAME = "newgame"
  """
  | Starts a game. The game type and state depend on the command arguments.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#newgame.
  """
  VALIDMOVES = "validmoves"
  """
  | Displays a list of every valid move in the current game.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#validmoves.
  """
  BESTMOVE = "bestmove"
  """
  | Search for the best move for the current game.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#bestmove.
  """
  PLAY = "play"
  """
  | Plays the specified MoveString in the current game.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#play.
  """
  PASS = "pass"
  """
  | Plays a passing move in the current game.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#pass.
  """
  UNDO = "undo"
  """
  | Undoes the specified amount of moves in the current game.
  | See https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol#undo.
  """
  EXIT = "exit"
  """
  Exits the engine.
  """

class Option(StrEnum):
  """
  Engine option.
  """
  STRATEGY_WHITE = "StrategyWhite"
  """
  AI strategy for player white.
  """
  STRATEGY_BLACK = "StrategyBlack"
  """
  AI strategy for player black.
  """
  MAX_BRANCHING_FACTOR = "MaxBranchingFactor"
  """
  Maximum amount of moves that can be considered for each node by Negamax agents.
  """
  NUM_THREADS = "NumThreads"
  """
  Available threads to parallelize AI thinking.
  """

class OptionType(StrEnum):
  """
  Option type.
  """
  BOOL = "bool"
  """
  Boolean type.
  """
  INT = "int"
  """
  Integer type.
  """
  FLOAT = "double"
  """
  Decimal type.
  """
  ENUM = "enum"
  """
  Enum type.
  """

class Strategy(StrEnum):
  """
  Possible AI strategy.
  """
  RANDOM = "Random"
  """
  Random strategy.
  """
  NEGAMAX = "Negamax"
  """
  Negamax with alpha-beta pruning strategy.
  """

class PlayerColor(StrEnum):
  """
  Player color.
  """
  WHITE = "White"
  """
  White.
  """
  BLACK = "Black"
  """
  Black.
  """

  @property
  def code(self) -> str:
    """
    Short string representation.

    :rtype: str
    """
    return self[0].lower()

  @property
  def opposite(self):
    """
    Opposite color.

    :rtype: PlayerColor
    """
    return _EnumUtils.PLAYER_COLOR_OPPOSITES[self]

class GameState(StrEnum):
  """
  Game state.
  """
  NOT_STARTED = "NotStarted"
  """
  Game not started yet.
  """
  IN_PROGRESS = "InProgress"
  """
  Game in progress.
  """
  DRAW = "Draw"
  """
  Game ended with a draw.
  """
  WHITE_WINS = "WhiteWins"
  """
  Game ended with victory for white.
  """
  BLACK_WINS = "BlackWins"
  """
  Game ended with victory for black.
  """

  @classmethod
  def parse(cls, state: str):
    """
    Parses a GameStateString.

    :param state: GameStateString.
    :type state: str
    :return: GameState.
    :rtype: GameState
    """
    return GameState(state) if state else GameState.NOT_STARTED

class GameType(Flag):
  """
  | Game type.
  | Determines the available pieces.
  """
  BASE = auto()
  """
  Base GameType, without expansions.
  """
  M = auto()
  """
  Mosquito GameType, with the Mosquito expansion.
  """
  L = auto()
  """
  Ladybug GameType, with the Ladybug expansion.
  """
  P = auto()
  """
  Pillbug GameType, with the Pillbug expansion.
  """

  @classmethod
  def parse(cls, game_type: str):
    """
    | Parses a GameTypeString.
    | The GameTypeString always needs to include the Base GameType.

    :param type: GameTypeString.
    :type type: str
    :raises ValueError: If it's not a valid GameTypeString.
    :return: GameType.
    :rtype: GameType
    """
    if game_type:
      base, *expansions = game_type.upper().split("+")
      try:
        if GameType[base] != GameType.BASE or expansions == [""] or len(expansions) > 1 and game_type.find("+") >= 0: raise KeyError()
        return reduce(lambda type, expansion: type | expansion, [GameType[expansion] for expansion in (expansions[0] if expansions else "")], GameType[base])
      except KeyError as e:
        raise ValueError(f"'{game_type}' is not a valid GameType") from e
    return GameType.BASE

  @property
  def tag(self) -> str:
    return (self.name or "").title()

  @property
  def index(self) -> int:
    subset = [flag for flag in GameType if flag != GameType.BASE]
    index = sum(1 << i for i, flag in enumerate(subset) if flag in self)
    return index

  def __str__(self) -> str:
    return "".join(gametype.tag + ("+" if gametype is GameType.BASE and len(self) > 1 else "") for gametype in self)

  def __repr__(self):
    return self.__str__()

class BugType(StrEnum):
  """
  Bug type.
  """
  QUEEN_BEE = "Q"
  """
  | Queen Bee.
  | Moves by 1 cell.
  | Must be placed after the first turn and by the fourth.
  | Surrounding this piece with other pieces is the win condition.
  """
  SPIDER = "S"
  """
  | Spider.
  | Moves by exactly 3 cells around the hive.
  """
  BEETLE = "B"
  """
  | Beetle.
  | Moves by 1 cell, but can also go above the hive.
  | Pieces below the Beetle cannot move and the cell is now considered of the Beetle's color.
  | Can go above other pieces already above the hive, piling up.
  """
  GRASSHOPPER = "G"
  """
  | Grasshopper.
  | Moves in a straight by jumping over other pieces.
  """
  SOLDIER_ANT = "A"
  """
  | Soldier Ant.
  | Moves by any amount of cells around the hive.
  """
  MOSQUITO = "M"
  """
  | Mosquito.
  | Available with GameType expansion M.
  | Moves by coping adiacent pieces' moveset.
  | If it goes above the hive by coping a Beetle, it can keep moving like a Beetle until it goes back down.
  | If it's only neighboring piece is another Mosquito, it cannot move.
  """
  LADYBUG = "L"
  """
  | Ladybug.
  | Available with GameType expansion L.
  | Moves by exactly 3 cells, like a Spider, but the first 2 steps must be above the hive.
  """
  PILLBUG = "P"
  """
  | Pillbug.
  | Available with GameType expansion P.
  | Moves by 1 cell.
  | Instead of moving itself, it can move a neighboring piece above itself, and then back down on a neighboring empty cell.
  """

class Direction(StrEnum):
  """
  | Hexagonal multi-layered grid direction.
  | The grid is assumed to be oriented with cells point-up.
  """
  RIGHT = "|-"
  """
  Right side of the hex: bug-
  """
  UP_RIGHT = "|/"
  """
  Up right side of the hex: bug/
  """
  UP_LEFT = "\\|"
  """
  Up left side of the hex: \\bug
  """
  LEFT = "-|"
  """
  Left side of the hex: -bug
  """
  DOWN_LEFT = "/|"
  """
  Down left side of the hex: /bug
  """
  DOWN_RIGHT = "|\\"
  """
  Down right side of the hex: bug\\
  """

  @classmethod
  def lefts(cls):
    """
    Returns all left directions. 

    :return: List of left directions.
    :rtype: list[Direction]
    """
    return _EnumUtils.DIRECTION_LEFTS

  @classmethod
  def rights(cls):
    """
    Returns all right directions. 

    :return: List of right directions.
    :rtype: list[Direction]
    """
    return _EnumUtils.DIRECTION_RIGHTS

  def __str__(self) -> str:
    return self.replace("|", "")

  def __repr__(self):
    return self.__str__()

  @property
  def opposite(self):
    """
    Opposite direction.

    :rtype: Direction
    """
    return _EnumUtils.DIRECTION_OPPOSITES[self]

  @property
  def clockwise(self):
    """
    Direction to the right (clockwise).

    :rtype: Direction
    """
    return _EnumUtils.DIRECTION_CLOCKWISES[self]

  @property
  def anticlockwise(self):
    """
    Direction to the left (anticlockwise).

    :rtype: Direction
    """
    return _EnumUtils.DIRECTION_ANTICLOCKWISES[self]

  @property
  def delta_index(self) -> int:
    """
    Direction index.

    :rtype: int
    """
    return _EnumUtils.DIRECTION_DELTA_INDICES[self]

  @property
  def is_right(self) -> bool:
    """
    Whether it's a right direction.

    :rtype: bool
    """
    return self in Direction.rights()

  @property
  def is_left(self) -> bool:
    """
    Whether it's a left direction.

    :rtype: bool
    """
    return self in Direction.lefts()

class _EnumUtils:
  """
  Internal utility class to make some enum methods and properties run in constant time.
  """

  PLAYER_COLOR_OPPOSITES: Final[dict[PlayerColor, PlayerColor]] = {color: PlayerColor.BLACK if color is PlayerColor.WHITE else PlayerColor.WHITE for color in PlayerColor}
  """
  Map of player colors to their opposite.
  """
  DIRECTION_LEFTS: Final[list[Direction]] = [direction for direction in Direction if direction is Direction.LEFT or direction is Direction.UP_LEFT or direction is Direction.DOWN_LEFT]
  """
  List of left directions.
  """
  DIRECTION_RIGHTS: Final[list[Direction]] = [direction for direction in Direction if direction is Direction.RIGHT or direction is Direction.UP_RIGHT or direction is Direction.DOWN_RIGHT]
  """
  List of right directions.
  """
  DIRECTION_DELTA_INDICES: Final[dict[Direction, int]] = {direction: list(Direction).index(direction) for direction in Direction}
  """
  Map of directions to their indices.
  """
  DIRECTION_OPPOSITES: Final[dict[Direction, Direction]] = {direction: list(Direction)[(list(Direction).index(direction) + 3) % 6] for direction in list(Direction)}
  """
  Map of directions to their opposite.
  """
  DIRECTION_CLOCKWISES: Final[dict[Direction, Direction]] = {direction: list(Direction)[(list(Direction).index(direction) + 5) % 6] for direction in list(Direction)}
  """
  Map of directions to their clockwise (right) neighboring direction.
  """
  DIRECTION_ANTICLOCKWISES: Final[dict[Direction, Direction]] = {direction: list(Direction)[(list(Direction).index(direction) + 1) % 6] for direction in list(Direction)}
  """
  Map of directions to their anitclockwise (left) neighboring direction.
  """
