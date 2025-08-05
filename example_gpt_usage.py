#!/usr/bin/env python3
"""
Example usage script for HiveGPT brain.

This script demonstrates how to integrate and use the GPT-based AI agent in the Hive engine.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.board import Board
from core.enums import GameState
from ai.brain import Random, AlphaBetaPruner

# Try to import GPTBrain, handling the case where dependencies might not be installed
try:
    from ai.gpt_brain import GPTBrain
    GPT_AVAILABLE = True
except ImportError as e:
    print(f"GPT brain not available: {e}")
    print("To use GPT brain, install: pip install torch transformers numpy datasets accelerate")
    GPT_AVAILABLE = False


def test_gpt_brain():
    """Test the GPT brain implementation."""
    if not GPT_AVAILABLE:
        print("GPT brain not available for testing")
        return
    
    print("Testing GPT Brain...")
    
    # Create a simple game board
    board = Board()
    
    try:
        # Initialize GPT brain (will use default config if no model is loaded)
        gpt_brain = GPTBrain()
        print("GPT Brain initialized successfully")
        
        # Test finding a move
        print("Testing move prediction...")
        best_move = gpt_brain.find_best_move(board, max_branching_factor=10)
        print(f"GPT Brain suggests move: {best_move}")
        
        # Test position evaluation
        print("Testing position evaluation...")
        evaluation = gpt_brain.evaluate_position(board)
        print(f"Position evaluation: {evaluation:.3f}")
        
    except Exception as e:
        print(f"Error testing GPT brain: {e}")


def compare_agents():
    """Compare different AI agents on the same position."""
    print("\nComparing AI agents...")
    
    # Create a simple game position
    board = Board()
    board.play("wS1")
    board.play("bS1 wS1/")
    board.play("wQ wS1-")
    
    print(f"Current position: {board}")
    print(f"Valid moves: {board.valid_moves}")
    
    # Test Random agent
    random_agent = Random()
    random_move = random_agent.find_best_move(board, max_branching_factor=10)
    print(f"Random agent move: {random_move}")
    
    # Test AlphaBeta agent
    ab_agent = AlphaBetaPruner()
    ab_move = ab_agent.find_best_move(board, max_branching_factor=10, max_depth=3)
    print(f"AlphaBeta agent move: {ab_move}")
    
    # Test GPT agent if available
    if GPT_AVAILABLE:
        try:
            gpt_agent = GPTBrain()
            gpt_move = gpt_agent.find_best_move(board, max_branching_factor=10)
            print(f"GPT agent move: {gpt_move}")
        except Exception as e:
            print(f"GPT agent error: {e}")


def play_game_with_gpt():
    """Play a short game using the GPT brain."""
    if not GPT_AVAILABLE:
        print("GPT brain not available for game play")
        return
    
    print("\nPlaying a game with GPT brain...")
    
    board = Board()
    gpt_brain = GPTBrain()
    random_brain = Random()
    
    move_count = 0
    max_moves = 20  # Limit game length for demo
    
    while not board.gameover and move_count < max_moves:
        print(f"\nMove {move_count + 1}: {board.current_player_color}'s turn")
        print(f"Current board: {board}")
        
        if board.current_player_color.name == "WHITE":
            # GPT plays white
            try:
                move = gpt_brain.find_best_move(board, max_branching_factor=8)
                print(f"GPT (White) plays: {move}")
            except Exception as e:
                print(f"GPT error, using random move: {e}")
                move = random_brain.find_best_move(board, max_branching_factor=8)
        else:
            # Random plays black
            move = random_brain.find_best_move(board, max_branching_factor=8)
            print(f"Random (Black) plays: {move}")
        
        board.play(move)
        move_count += 1
    
    print(f"\nFinal position: {board}")
    print(f"Game state: {board.state}")


def main():
    """Main function demonstrating GPT brain usage."""
    print("HiveGPT Brain Example")
    print("=" * 40)
    
    # Test GPT brain basic functionality
    test_gpt_brain()
    
    # Compare different agents
    compare_agents()
    
    # Play a demo game
    play_game_with_gpt()
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()
