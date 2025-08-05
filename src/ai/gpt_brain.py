"""
GPT-based AI agent for Hive game engine.

This module implements a GPT-based brain that can be fine-tuned on Hive game data
to learn strategic gameplay similar to the ALLIE chess approach but adapted for Hive.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import ModelOutput
import numpy as np
from pathlib import Path
import pickle

from core.board import Board
from core.game import Move, Bug, Position
from core.enums import GameState, PlayerColor, BugType
from ai.brain import Brain


@dataclass
class HiveGPTOutput(ModelOutput):
    """
    Output class for HiveGPT model containing both policy and value predictions.
    """
    loss: Optional[torch.FloatTensor] = None
    policy_logits: torch.FloatTensor = None
    value_prediction: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HiveGameEncoder:
    """
    Encoder for converting Hive game states and moves into sequences suitable for GPT training.
    """
    
    def __init__(self):
        # Special tokens
        self.GAME_START = "<GAME_START>"
        self.MOVE_SEP = "<MOVE>"
        self.PLAYER_WHITE = "<WHITE>"
        self.PLAYER_BLACK = "<BLACK>"
        self.GAME_END = "<GAME_END>"
        self.WIN_WHITE = "<WIN_WHITE>"
        self.WIN_BLACK = "<WIN_BLACK>"
        self.DRAW = "<DRAW>"
        self.PAD_TOKEN = "<PAD>"
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary for Hive game representation."""
        vocab = [
            self.PAD_TOKEN,
            self.GAME_START, 
            self.MOVE_SEP,
            self.PLAYER_WHITE,
            self.PLAYER_BLACK,
            self.GAME_END,
            self.WIN_WHITE,
            self.WIN_BLACK,
            self.DRAW,
            "pass"
        ]
        
        # Add all possible bug pieces
        for color in PlayerColor:
            for bug_type in BugType:
                for bug_id in [None, 1, 2, 3]:
                    if bug_type in [BugType.ANT, BugType.BEETLE, BugType.GRASSHOPPER] and bug_id is None:
                        continue
                    if bug_type == BugType.QUEEN and bug_id is not None:
                        continue
                    if bug_type in [BugType.SPIDER, BugType.LADYBUG, BugType.MOSQUITO, BugType.PILLBUG] and bug_id is not None:
                        continue
                        
                    bug_str = f"{color.code}{bug_type}"
                    if bug_id is not None:
                        bug_str += str(bug_id)
                    vocab.append(bug_str)
        
        # Add position indicators (relative positioning)
        directions = ["/", "\\", "-", "|"]
        vocab.extend(directions)
        
        # Add positional tokens for common board positions
        for q in range(-10, 11):
            for r in range(-10, 11):
                if abs(q) + abs(r) <= 10:  # Limit to reasonable board size
                    vocab.append(f"@{q},{r}")
        
        return vocab
    
    def encode_game_sequence(self, move_strings: List[str], game_result: str) -> List[int]:
        """
        Encode a complete game sequence into token IDs.
        
        Args:
            move_strings: List of move strings from the game
            game_result: Game result ("white", "black", "draw")
            
        Returns:
            List of token IDs representing the game sequence
        """
        tokens = [self.GAME_START]
        
        current_player = PlayerColor.WHITE
        for move_str in move_strings:
            # Add player indicator
            player_token = self.PLAYER_WHITE if current_player == PlayerColor.WHITE else self.PLAYER_BLACK
            tokens.append(player_token)
            tokens.append(self.MOVE_SEP)
            
            # Add move
            if move_str.strip().lower() == "pass":
                tokens.append("pass")
            else:
                tokens.append(move_str.strip())
            
            current_player = current_player.opposite
        
        # Add game ending
        tokens.append(self.GAME_END)
        if game_result.lower() == "white":
            tokens.append(self.WIN_WHITE)
        elif game_result.lower() == "black":
            tokens.append(self.WIN_BLACK)
        else:
            tokens.append(self.DRAW)
        
        return [self.token_to_id.get(token, self.token_to_id[self.PAD_TOKEN]) for token in tokens]
    
    def decode_move(self, token_id: int) -> str:
        """Decode a token ID back to a move string."""
        return self.id_to_token.get(token_id, self.PAD_TOKEN)


class HiveGPTModel(nn.Module):
    """
    GPT-based model for Hive gameplay with policy and value heads.
    Adapted from the ALLIE architecture without the time/pondering component.
    """
    
    def __init__(self, config: GPT2Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Use GPT2 as backbone but with custom embedding size
        self.transformer = GPT2Model(config)
        
        # Resize embeddings to match our vocabulary
        self.transformer.resize_token_embeddings(vocab_size)
        
        # Policy head - predicts next move probabilities
        self.policy_head = nn.Linear(config.hidden_size, vocab_size)
        
        # Value head - predicts game outcome from current position
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Tanh()  # Squash to [-1, 1] range
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        values: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> HiveGPTOutput:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            labels: Target token IDs for policy loss
            values: Target values for value loss
            
        Returns:
            HiveGPTOutput containing loss, policy logits, and value predictions
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        
        # Policy head
        policy_logits = self.policy_head(hidden_states)
        
        # Value head - use last token's representation
        value_prediction = self.value_head(hidden_states[:, -1, :])
        
        loss = None
        if labels is not None or values is not None:
            loss = 0.0
            
            # Policy loss (cross-entropy)
            if labels is not None:
                shift_logits = policy_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                policy_loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss += policy_loss
            
            # Value loss (MSE)
            if values is not None:
                value_loss = F.mse_loss(value_prediction.squeeze(-1), values)
                loss += value_loss
        
        return HiveGPTOutput(
            loss=loss,
            policy_logits=policy_logits,
            value_prediction=value_prediction,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPTBrain(Brain):
    """
    GPT-based AI agent for Hive.
    
    This brain uses a fine-tuned GPT model to predict moves and evaluate positions,
    similar to the ALLIE approach for chess but adapted for Hive.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        super().__init__()
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize encoder
        self.encoder = HiveGameEncoder()
        
        # Initialize model
        config = GPT2Config(
            vocab_size=self.encoder.vocab_size,
            n_positions=1024,  # Max sequence length
            n_embd=768,        # Hidden size  
            n_layer=12,        # Number of layers
            n_head=12,         # Number of attention heads
            n_inner=3072,      # FFN inner dimension
        )
        
        self.model = HiveGPTModel(config, self.encoder.vocab_size)
        
        # Load pre-trained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _find_best_move(self, board: Board, max_branching_factor: int, max_depth: int = 0, time_limit: int = 0) -> str:
        """
        Find the best move using the GPT model.
        
        Args:
            board: Current game board
            max_branching_factor: Maximum number of moves to consider
            max_depth: Ignored for GPT brain
            time_limit: Ignored for GPT brain
            
        Returns:
            Best move as a string
        """
        # Encode current game state
        game_sequence = self._board_to_sequence(board)
        input_ids = torch.tensor([game_sequence], dtype=torch.long, device=self.device)
        
        # Get valid moves
        valid_moves = list(board.calculate_valid_moves())
        if not valid_moves:
            return Move.PASS
        
        # Limit to max branching factor
        if len(valid_moves) > max_branching_factor > 0:
            valid_moves = valid_moves[:max_branching_factor]
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids)
            policy_logits = outputs.policy_logits[0, -1, :]  # Last token predictions
            
            # Convert moves to tokens and get their probabilities
            move_scores = []
            for move in valid_moves:
                move_str = board.stringify_move(move)
                token_id = self.encoder.token_to_id.get(move_str, self.encoder.token_to_id[self.encoder.PAD_TOKEN])
                score = policy_logits[token_id].item()
                move_scores.append((move, score))
            
            # Sort by score and return best move
            move_scores.sort(key=lambda x: x[1], reverse=True)
            best_move = move_scores[0][0]
            
            return board.stringify_move(best_move)
    
    def _board_to_sequence(self, board: Board) -> List[int]:
        """
        Convert board state to token sequence.
        
        Args:
            board: Current game board
            
        Returns:
            List of token IDs representing the game state
        """
        # Use the move history to reconstruct sequence
        move_strings = board.move_strings
        
        tokens = [self.encoder.token_to_id[self.encoder.GAME_START]]
        
        current_player = PlayerColor.WHITE
        for move_str in move_strings:
            # Add player indicator
            player_token = self.encoder.PLAYER_WHITE if current_player == PlayerColor.WHITE else self.encoder.PLAYER_BLACK
            tokens.append(self.encoder.token_to_id[player_token])
            tokens.append(self.encoder.token_to_id[self.encoder.MOVE_SEP])
            
            # Add move
            move_token_id = self.encoder.token_to_id.get(move_str, self.encoder.token_to_id[self.encoder.PAD_TOKEN])
            tokens.append(move_token_id)
            
            current_player = current_player.opposite
        
        # Add current player indicator for next move
        player_token = self.encoder.PLAYER_WHITE if board.current_player_color == PlayerColor.WHITE else self.encoder.PLAYER_BLACK
        tokens.append(self.encoder.token_to_id[player_token])
        tokens.append(self.encoder.token_to_id[self.encoder.MOVE_SEP])
        
        return tokens
    
    def evaluate_position(self, board: Board) -> float:
        """
        Evaluate the current position using the value head.
        
        Args:
            board: Current game board
            
        Returns:
            Position evaluation in range [-1, 1]
        """
        game_sequence = self._board_to_sequence(board)
        input_ids = torch.tensor([game_sequence], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            value = outputs.value_prediction[0].item()
            
        return value
    
    def save_model(self, path: str):
        """Save the model to disk."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'encoder': self.encoder,
        }
        torch.save(save_dict, path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'encoder' in checkpoint:
            self.encoder = checkpoint['encoder']


class HiveDataset(torch.utils.data.Dataset):
    """
    Dataset class for training the GPT model on Hive games.
    """
    
    def __init__(self, games_data: List[Dict[str, Any]], encoder: HiveGameEncoder, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            games_data: List of game dictionaries with 'moves' and 'result' keys
            encoder: HiveGameEncoder instance
            max_length: Maximum sequence length
        """
        self.games_data = games_data
        self.encoder = encoder
        self.max_length = max_length
        
        # Preprocess games into sequences
        self.sequences = []
        self.values = []
        
        for game in games_data:
            moves = game['moves']
            result = game['result']
            
            # Convert result to numerical value
            if result.lower() == 'white':
                value = 1.0
            elif result.lower() == 'black':
                value = -1.0
            else:
                value = 0.0
            
            # Encode sequence
            sequence = encoder.encode_game_sequence(moves, result)
            
            # Truncate if too long
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            
            self.sequences.append(sequence)
            self.values.append(value)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        value = self.values[idx]
        
        # Pad sequence
        padded_sequence = sequence + [self.encoder.token_to_id[self.encoder.PAD_TOKEN]] * (self.max_length - len(sequence))
        
        return {
            'input_ids': torch.tensor(padded_sequence[:-1], dtype=torch.long),
            'labels': torch.tensor(padded_sequence[1:], dtype=torch.long),
            'values': torch.tensor(value, dtype=torch.float),
        }


class HiveGPTTrainer:
    """
    Trainer class for fine-tuning the GPT model on Hive game data.
    """
    
    def __init__(self, model_save_path: str = "hive_gpt_model.pt"):
        self.model_save_path = model_save_path
        self.encoder = HiveGameEncoder()
        
        # Initialize model
        config = GPT2Config(
            vocab_size=self.encoder.vocab_size,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072,
        )
        
        self.model = HiveGPTModel(config, self.encoder.vocab_size)
        
        # Load pre-trained GPT-2 weights (excluding embeddings)
        try:
            gpt2_model = GPT2Model.from_pretrained('gpt2-medium')
            self._transfer_weights(gpt2_model)
            print("Loaded pre-trained GPT-2 medium weights")
        except Exception as e:
            print(f"Could not load GPT-2 weights: {e}")
            print("Training from scratch")
    
    def _transfer_weights(self, gpt2_model):
        """Transfer weights from pre-trained GPT-2 model (excluding embeddings)."""
        # Copy transformer weights but not embeddings
        with torch.no_grad():
            for name, param in gpt2_model.named_parameters():
                if 'wte' not in name and 'wpe' not in name:  # Skip word/position embeddings
                    if hasattr(self.model.transformer, name.split('.')[0]):
                        target_param = self.model.transformer
                        for attr in name.split('.'):
                            target_param = getattr(target_param, attr)
                        if target_param.shape == param.shape:
                            target_param.copy_(param)
    
    def train(self, games_data: List[Dict[str, Any]], 
              output_dir: str = "./hive_gpt_output",
              num_epochs: int = 3,
              batch_size: int = 8,
              learning_rate: float = 5e-5,
              warmup_steps: int = 500):
        """
        Train the model on game data.
        
        Args:
            games_data: List of game dictionaries
            output_dir: Directory to save training outputs
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
        """
        # Create dataset
        dataset = HiveDataset(games_data, self.encoder)
        
        # Split into train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            learning_rate=learning_rate,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        self.save_model(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
    
    def save_model(self, path: str):
        """Save the trained model."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'encoder': self.encoder,
        }
        torch.save(save_dict, path)


def load_game_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load game data from file.
    
    Expected format: JSON file with list of games, each containing:
    - 'moves': List of move strings
    - 'result': Game result ('white', 'black', or 'draw')
    
    Args:
        file_path: Path to the game data file
        
    Returns:
        List of game dictionaries
    """
    with open(file_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Example usage for training
    print("HiveGPT Brain - GPT-based AI for Hive")
    print("This module provides a GPT-based AI agent that can be fine-tuned on Hive game data.")
    print("\nTo use:")
    print("1. Prepare game data in JSON format")
    print("2. Create HiveGPTTrainer and call train()")
    print("3. Use GPTBrain with the trained model")
