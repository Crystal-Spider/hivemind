import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import numpy as np
from copy import deepcopy
from time import time
from core.enums import GameType, GameState, PlayerColor, BugType, Direction
from core.board import Board  # type: ignore
from core.enums import GameState, PlayerColor  # type: ignore
from ai.alphamcts import AlphaMCTS  # type: ignore
from ai.resnet import ResNet  # type: ignore


def compute_value(winner, s: Board) -> float:
    """
    Calcola il value target: se lo stato s è a favore del vincitore allora 1, altrimenti -1.
    """
    if s.current_player_color in winner:  # type: ignore
        return 1.0
    elif s.current_player_color in winner:
        return -1.0
    return 0.0
def data_augmentation(sample):
    """
    Esegue data augmentation sul sample.
    In questo esempio non viene effettuata alcuna trasformazione e si restituisce il sample in una lista.
    """
    return [sample]

class Player:
    def __init__(self, network: ResNet):
        self.network = network
        self.mcts = AlphaMCTS(self.network)
    def select_move(self, state: Board, time_limit: int = 1) -> str:
        return self.mcts._find_best_move(state, time_limit=time_limit)  # type: ignore

def duel(player1: Player, player2: Player) -> bool:
    """
    Duello tra player1 e player2.
    Restituisce True se player1 vince.
    """
    s = Board()  # Inizializza una nuova partita
    current_player = player1
    while s.state == GameState.IN_PROGRESS:  # type: ignore
        move = current_player.select_move(s, time_limit=1)
        s.play(move)  # type: ignore
        current_player = player2 if current_player == player1 else player1
    return s.state == GameState.WHITE_WINS  # type: ignore

def copy_network(net: ResNet) -> ResNet:
    """
    Effettua una copia profonda della rete neurale.
    """
    new_net = ResNet()  # type: ignore
    new_net.load_state_dict(deepcopy(net.state_dict()))
    return new_net

def training_phase(network: ResNet, training_data: list, lr: float = 0.001, epochs: int = 1) -> ResNet:
    """
    Allenamento della rete sul training_data per aggiornare sia la policy che il valore.
    """
    network.train()  # Imposta la rete in modalità di allenamento
    optimizer = optim.Adam(network.parameters(), lr=lr)  # type: ignore
    loss_fn_value = nn.MSELoss()  # type: ignore
    loss_fn_policy = nn.CrossEntropyLoss()  # type: ignore

    for epoch in range(epochs):
        epoch_loss_value = 0.0
        epoch_loss_policy = 0.0
        for sample in training_data:
            s, pi, value = sample
            state_tensor = torch.tensor(s.get_board_matrix(mode=0), dtype=torch.float32).unsqueeze(0)  # type: ignore
            pi_tensor = torch.tensor(pi, dtype=torch.float32).unsqueeze(0)  # type: ignore
            value_tensor = torch.tensor(value, dtype=torch.float32).unsqueeze(0)  # type: ignore

            policy_pred, value_pred = network(state_tensor)
            loss_value = loss_fn_value(value_pred.squeeze(), value_tensor.squeeze())  # type: ignore
            loss_policy = loss_fn_policy(policy_pred, pi_tensor.argmax(dim=1))  # type: ignore
            loss = loss_value + loss_policy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_value += loss_value.item()
            epoch_loss_policy += loss_policy.item()

        print(f"Training epoch {epoch+1}, Value Loss: {epoch_loss_value:.4f}, Policy Loss: {epoch_loss_policy:.4f}")
    return network


# --- Main loop ---
if __name__ == "__main__":
    number_of_iterations = 1
    number_of_games = 1

    # Initialization
    f_theta = ResNet()  # type: ignore
    Tx = []

    for iteration in range(number_of_iterations):
        print(f"\n=== Iteration {iteration+1}/{number_of_iterations} ===")
        # Self-play
        for game in range(number_of_games):
            s = Board()  # Inizializza una nuova partita
            T_game = []
            mcts_game = AlphaMCTS()  # type: ignore
            turn = 0
            winner = None
        
            while s.state == GameState.IN_PROGRESS or s.state==GameState.NOT_STARTED:  # type: ignore
                turn += 1
                best_move, pi = mcts_game._find_best_move(s, time_limit=0.1)  # type: ignore
                
                T_game.append((s, pi))
                a = best_move
                print(a)
                s = s.play(a)  # type: ignore
                if s.state != GameState.IN_PROGRESS:  # type: ignore
                    winner = s.state

            for sample in T_game:
                s_sample, pi = sample
                value = compute_value(winner, s_sample)
                Tx += data_augmentation((s_sample, pi, value))
                

        # Training phase
        f_theta_new = copy_network(f_theta)
        f_theta_new = training_phase(f_theta_new, Tx, lr=0.001, epochs=1)

        # Evaluation phase
        new_player = Player(f_theta_new)
        old_player = Player(f_theta)
        new_player_is_stronger = duel(new_player, old_player)
        if new_player_is_stronger:
            print("La nuova rete è più forte. Aggiorno il modello!")
            f_theta = copy_network(f_theta_new)
            torch.save(f_theta.state_dict(), "resnet_weights/f_theta.pth")
        else:
            print("Il modello corrente rimane invariato.")
