import math
from typing import Optional
from random import choice
from time import time
from copy import deepcopy
from core.board import Board
from core.enums import PlayerColor, GameState
from core.board import Move
from ai.brain import Brain
import torch
from ai.resnet import ResNet
class AlphaMCTSNode:
    def __init__(self, board: Board, parent: Optional['AlphaMCTSNode'] = None, move: Optional[Move] = None):
        self.board = board
        self.parent = parent
        self.move: Optional[Move] = move
        self.children: list[AlphaMCTSNode] = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.board.calculate_valid_moves_for_player(self.board.current_player_color))

    def get_ucb(self, child: 'AlphaMCTSNode', exploration_weight: float = 1.41) -> float:
        if child.visits == 0:
            return 0
        q= 1- ((child.wins/child.visits)+1)/2
        return q + exploration_weight * math.sqrt(math.log( self.visits) / child.visits)

    def best_child(self, exploration_weight: float = 1.41) -> 'AlphaMCTSNode':
        choices_weights = [self.get_ucb(child, exploration_weight) for child in self.children]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self,policy):

        moves=self.board.calculate_valid_moves_for_player(self.board.current_player_color)
        for move in moves:
            position=self.board.move_to_index(self.board.stringify_move(move))
            if policy[position[0],position[1],position[2]]>0:
                new_board = deepcopy(self.board)
                new_board.play(self.board.stringify_move(move))
                child_node = AlphaMCTSNode(new_board, self, move)
                self.children.append(child_node)
        return

    def backpropagate(self, result:int):
        self.visits += 1
        self.wins += result

        if self.parent:
            self.parent.backpropagate(1-result)

class AlphaMCTS(Brain):
    def __init__(self):
        super().__init__()
        self.model=ResNet()
        self.size=(7,14,14)

    def _find_best_move(self, board: Board, max_branching_factor: int = 0,
                        max_depth: int = 0, time_limit: int = 0) -> str:
        end_time = time() + time_limit
        root = AlphaMCTSNode(board)
        count=0
        while time() < end_time:
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child(exploration_weight=1.41)

            if not node.is_fully_expanded():

                policy,value=self.model(
                    torch.tensor(node.board.get_board_matrix(mode=0)).unsqueeze(0).float()
                )
                policy=policy.squeeze().cpu().detach().numpy()
                policy=policy.reshape(self.size)
                node.expand(policy)
                result=value.item()
            else:
                result = node.board.state
            node.backpropagate(result)
            count+=1
        
        
        print("Simulated nodes: ",count)
        best_move = root.best_child(0).move
        return board.stringify_move(best_move) 
