#!/usr/bin/env python3
"""
Training script for HiveGPT model.

This script demonstrates how to train the GPT-based AI on Hive game data.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from ai.gpt_brain import HiveGPTTrainer, load_game_data


def generate_sample_data() -> List[Dict[str, Any]]:
    """
    Generate sample training data for demonstration.
    In practice, this would be replaced with real game data.
    """
    sample_games = [
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wA1 wQ/", "bA1 bQ/"],
            "result": "white"
        },
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wA1 wQ/", "bA1 bQ/", "wG1 wA1/"],
            "result": "black"
        },
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "pass", "pass"],
            "result": "draw"
        }
    ]
    
    # Replicate sample data to have more training examples
    training_data = []
    for _ in range(100):  # Create 300 total samples
        training_data.extend(sample_games)
    
    return training_data


def main():
    parser = argparse.ArgumentParser(description="Train HiveGPT model")
    parser.add_argument("--data-file", type=str, help="Path to game data JSON file")
    parser.add_argument("--output-dir", type=str, default="./hive_gpt_output", 
                       help="Output directory for training")
    parser.add_argument("--model-save-path", type=str, default="hive_gpt_model.pt",
                       help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use-sample-data", action="store_true", 
                       help="Use generated sample data instead of loading from file")
    
    args = parser.parse_args()
    
    # Load or generate training data
    if args.use_sample_data or not args.data_file:
        print("Using generated sample data...")
        games_data = generate_sample_data()
    else:
        print(f"Loading data from {args.data_file}...")
        games_data = load_game_data(args.data_file)
    
    print(f"Loaded {len(games_data)} games for training")
    
    # Initialize trainer
    trainer = HiveGPTTrainer(model_save_path=args.model_save_path)
    
    # Train the model
    print("Starting training...")
    trainer.train(
        games_data=games_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("Training completed!")
    print(f"Model saved to {args.model_save_path}")


if __name__ == "__main__":
    main()
