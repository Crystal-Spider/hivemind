import pytest
from core.enums import PlayerColor, GameState, GameType, Direction

class TestPlayerColor:
  def test_code(self):
    assert PlayerColor.WHITE.code == "w"
    assert PlayerColor.BLACK.code == "b"

  def test_opposite(self):
    assert PlayerColor.WHITE.opposite == PlayerColor.BLACK
    assert PlayerColor.BLACK.opposite == PlayerColor.WHITE

class TestGameState:
  def test_parse(self):
    assert GameState.parse("NotStarted") == GameState.NOT_STARTED
    assert GameState.parse("InProgress") == GameState.IN_PROGRESS
    assert GameState.parse("Draw") == GameState.DRAW
    assert GameState.parse("WhiteWins") == GameState.WHITE_WINS
    assert GameState.parse("BlackWins") == GameState.BLACK_WINS
    assert GameState.parse("") == GameState.NOT_STARTED

class TestGameType:
  def test_parse(self):
    assert GameType.parse("") == GameType.BASE
    assert GameType.parse("Base") == GameType.BASE
    assert GameType.parse("Base+M") == (GameType.BASE | GameType.M)
    assert GameType.parse("Base+L") == (GameType.BASE | GameType.L)
    assert GameType.parse("Base+P") == (GameType.BASE | GameType.P)
    assert GameType.parse("Base+ML") == (GameType.BASE | GameType.M | GameType.L)
    assert GameType.parse("Base+MP") == (GameType.BASE | GameType.M | GameType.P)
    assert GameType.parse("Base+LM") == (GameType.BASE | GameType.M | GameType.L)
    assert GameType.parse("Base+LP") == (GameType.BASE | GameType.L | GameType.P)
    assert GameType.parse("Base+PM") == (GameType.BASE | GameType.M | GameType.P)
    assert GameType.parse("Base+PL") == (GameType.BASE | GameType.L | GameType.P)
    assert GameType.parse("Base+MLP") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+MPL") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+LMP") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+LPM") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+PML") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    assert GameType.parse("Base+PLM") == (GameType.BASE | GameType.M | GameType.L | GameType.P)
    with pytest.raises(ValueError):
      GameType.parse("Invalid")
    with pytest.raises(ValueError):
      GameType.parse("Base+Invalid")
    with pytest.raises(ValueError):
      GameType.parse("M")
    with pytest.raises(ValueError):
      GameType.parse("L")
    with pytest.raises(ValueError):
      GameType.parse("P")

  def test_tag(self):
    assert GameType.BASE.tag == "Base"
    assert GameType.M.tag == "M"
    assert GameType.L.tag == "L"
    assert GameType.P.tag == "P"

  def test_str(self):
    assert str(GameType.BASE) == "Base"
    assert str(GameType.BASE | GameType.M) == "Base+M"
    assert str(GameType.BASE | GameType.L) == "Base+L"
    assert str(GameType.BASE | GameType.P) == "Base+P"
    assert str(GameType.BASE | GameType.M | GameType.L) == "Base+ML"
    assert str(GameType.BASE | GameType.M | GameType.P) == "Base+MP"
    assert str(GameType.BASE | GameType.L | GameType.M) == "Base+ML"
    assert str(GameType.BASE | GameType.L | GameType.P) == "Base+LP"
    assert str(GameType.BASE | GameType.P | GameType.M) == "Base+MP"
    assert str(GameType.BASE | GameType.P | GameType.L) == "Base+LP"
    assert str(GameType.BASE | GameType.M | GameType.L | GameType.P) == "Base+MLP"
    assert str(GameType.BASE | GameType.M | GameType.P | GameType.L) == "Base+MLP"
    assert str(GameType.BASE | GameType.L | GameType.M | GameType.P) == "Base+MLP"
    assert str(GameType.BASE | GameType.L | GameType.P | GameType.M) == "Base+MLP"
    assert str(GameType.BASE | GameType.P | GameType.M | GameType.L) == "Base+MLP"
    assert str(GameType.BASE | GameType.P | GameType.L | GameType.M) == "Base+MLP"

class TestDirection:
  def test_lefts(self):
    directions = Direction.lefts()
    assert Direction.RIGHT not in directions
    assert Direction.UP_RIGHT not in directions
    assert Direction.UP_LEFT in directions
    assert Direction.LEFT in directions
    assert Direction.DOWN_LEFT in directions
    assert Direction.DOWN_RIGHT not in directions

  def test_rights(self):
    directions = Direction.rights()
    assert Direction.RIGHT in directions
    assert Direction.UP_RIGHT in directions
    assert Direction.UP_LEFT not in directions
    assert Direction.LEFT not in directions
    assert Direction.DOWN_LEFT not in directions
    assert Direction.DOWN_RIGHT in directions

  def test_str(self):
    assert str(Direction.RIGHT) == "-"
    assert str(Direction.UP_RIGHT) == "/"
    assert str(Direction.UP_LEFT) == "\\"
    assert str(Direction.LEFT) == "-"
    assert str(Direction.DOWN_LEFT) == "/"
    assert str(Direction.DOWN_RIGHT) == "\\"

  def test_opposite(self):
    assert Direction.RIGHT.opposite == Direction.LEFT
    assert Direction.UP_RIGHT.opposite == Direction.DOWN_LEFT
    assert Direction.UP_LEFT.opposite == Direction.DOWN_RIGHT
    assert Direction.LEFT.opposite == Direction.RIGHT
    assert Direction.DOWN_LEFT.opposite == Direction.UP_RIGHT
    assert Direction.DOWN_RIGHT.opposite == Direction.UP_LEFT

  def test_left_of(self):
    assert Direction.RIGHT.anticlockwise == Direction.UP_RIGHT
    assert Direction.UP_RIGHT.anticlockwise == Direction.UP_LEFT
    assert Direction.UP_LEFT.anticlockwise == Direction.LEFT
    assert Direction.LEFT.anticlockwise == Direction.DOWN_LEFT
    assert Direction.DOWN_LEFT.anticlockwise == Direction.DOWN_RIGHT
    assert Direction.DOWN_RIGHT.anticlockwise == Direction.RIGHT

  def test_right_of(self):
    assert Direction.RIGHT.clockwise == Direction.DOWN_RIGHT
    assert Direction.UP_RIGHT.clockwise == Direction.RIGHT
    assert Direction.UP_LEFT.clockwise == Direction.UP_RIGHT
    assert Direction.LEFT.clockwise == Direction.UP_LEFT
    assert Direction.DOWN_LEFT.clockwise == Direction.LEFT
    assert Direction.DOWN_RIGHT.clockwise == Direction.DOWN_LEFT

  def test_delta_index(self):
    assert Direction.RIGHT.delta_index == 0
    assert Direction.UP_RIGHT.delta_index == 1
    assert Direction.UP_LEFT.delta_index == 2
    assert Direction.LEFT.delta_index == 3
    assert Direction.DOWN_LEFT.delta_index == 4
    assert Direction.DOWN_RIGHT.delta_index == 5

  def test_is_right(self):
    assert Direction.RIGHT.is_right
    assert Direction.UP_RIGHT.is_right
    assert Direction.DOWN_RIGHT.is_right
    assert not Direction.UP_LEFT.is_right
    assert not Direction.LEFT.is_right
    assert not Direction.DOWN_LEFT.is_right

  def test_is_left(self):
    assert not Direction.RIGHT.is_left
    assert not Direction.UP_RIGHT.is_left
    assert not Direction.DOWN_RIGHT.is_left
    assert Direction.UP_LEFT.is_left
    assert Direction.LEFT.is_left
    assert Direction.DOWN_LEFT.is_left

if __name__ == '__main__':
  pytest.main()
