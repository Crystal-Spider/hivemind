import math
from typing import Optional
from random import choice
from time import time
from copy import deepcopy
from core.board import Board
from core.enums import PlayerColor, GameState
from core.board import Move
from ai.brain import Brain
class MCTSNode:
    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None, move: Optional[Move] = None):
        self.board = board
        self.parent = parent
        self.move: Optional[Move] = move
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.board.calculate_valid_moves_for_player(self.board.current_player_color))

    def get_ucb(self, child: 'MCTSNode', exploration_weight: float = 1.41) -> float:
        q= 1- ((child.wins/child.visits)+1)/2
        return q + exploration_weight * math.sqrt(math.log( self.visits) / child.visits)

    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        choices_weights = [self.get_ucb(child, exploration_weight) for child in self.children]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        untried_moves = [move for move in self.board.calculate_valid_moves_for_player(self.board.current_player_color) if move not in [child.move for child in self.children]]
        move = choice(untried_moves)
        new_board = deepcopy(self.board)
        new_board.play(self.board.stringify_move(move))
        child_node = MCTSNode(new_board, self, move)
        self.children.append(child_node)
        return child_node

    def rollout(self, max_depth: int = 400) -> GameState:
        current_rollout_board = deepcopy(self.board)
        depth = 0
        while current_rollout_board.state == GameState.IN_PROGRESS and depth < max_depth:
            possible_moves = list(current_rollout_board.calculate_valid_moves_for_player(current_rollout_board.current_player_color))
            move = choice(possible_moves)
            current_rollout_board.play(self.board.stringify_move(move))
            depth += 1
        return current_rollout_board.state

    def backpropagate(self, result: GameState):
        self.visits += 1
        if result == GameState.WHITE_WINS and self.board.current_player_color == PlayerColor.WHITE:
            self.wins += 1
        elif result == GameState.BLACK_WINS and self.board.current_player_color == PlayerColor.BLACK:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

class MCTS(Brain):
    def __init__(self):
        super().__init__()

    def _find_best_move(self, board: Board, max_branching_factor: int = 0,
                        max_depth: int = 0, time_limit: int = 0) -> str:
        end_time = time() + time_limit
        root = MCTSNode(board)
        count=0
        while time() < end_time:
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child(exploration_weight=1.41)
            if not node.is_fully_expanded():
                node = node.expand()
            result = node.rollout(max_depth=max_depth)
            node.backpropagate(result)
            count+=1
        print("Simulated games: ",count)
        
        best_move = root.best_child(0).move
        return board.stringify_move(best_move)
