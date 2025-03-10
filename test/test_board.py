import pytest
from core.board import Board

class TestBoard:
  def test_hash(self):
    board = Board()
    assert board.hash() == 0
    board.play("wS1")
    hash1 = board.hash()
    assert hash1 != 0
    board.play("bS1 wS1-")
    hash2 = board.hash()
    assert hash2 != 0 and hash2 != hash1
    board.play("wQ -wS1")
    hash3 = board.hash()
    assert hash3 != 0 and hash3 != hash2 and hash3 != hash1
    board.play("bQ bS1-")
    hash4 = board.hash()
    assert hash4 != 0 and hash4 != hash3 and hash4 != hash2 and hash4 != hash1
    board.play("wG1 -wQ")
    hash5 = board.hash()
    assert hash5 != 0 and hash5 != hash4 and hash5 != hash3 and hash5 != hash2 and hash5 != hash1
    board.undo()
    hash6 = board.hash()
    assert hash6 == hash4
    board.play("wG1 -wQ")
    hash7 = board.hash()
    assert hash7 == hash5
    board.undo()
    hash8 = board.hash()
    assert hash8 == hash4
    board.undo()
    hash9 = board.hash()
    assert hash9 == hash3
    board.undo()
    hash10 = board.hash()
    assert hash10 == hash2
    board.undo()
    hash11 = board.hash()
    assert hash11 == hash1
    board.undo()
    hash12 = board.hash()
    assert hash12 == 0

if __name__ == "__main__":
  pytest.main()
