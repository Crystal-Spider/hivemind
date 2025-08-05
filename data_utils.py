#!/usr/bin/env python3
"""
Utility functions for HiveGPT data processing and game analysis.
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.board import Board
from core.enums import GameState, PlayerColor
from ai.gpt_brain import HiveGameEncoder


class HiveGameDataProcessor:
    """
    Processor for converting various Hive game data formats into training data for HiveGPT.
    """
    
    def __init__(self):
        self.encoder = HiveGameEncoder()
    
    def parse_gamestring_to_moves(self, gamestring: str) -> List[str]:
        """
        Parse a Hive gamestring into individual moves.
        
        Args:
            gamestring: Complete game string (e.g., "Base;InProgress;White[1];wS1;bS1 wS1/;...")
            
        Returns:
            List of move strings
        """
        parts = gamestring.split(';')
        if len(parts) < 4:
            return []
        
        # Skip the first 3 parts (game type, state, turn) and extract moves
        moves = parts[3:]
        return [move.strip() for move in moves if move.strip()]
    
    def extract_game_result(self, gamestring: str) -> str:
        """
        Extract game result from gamestring.
        
        Args:
            gamestring: Complete game string
            
        Returns:
            Game result: "white", "black", or "draw"
        """
        try:
            board = Board(gamestring)
            if board.state == GameState.WHITE_WINS:
                return "white"
            elif board.state == GameState.BLACK_WINS:
                return "black"
            elif board.state == GameState.DRAW:
                return "draw"
            else:
                # Game is still in progress, determine winner by current state
                if board.current_player_has_won:
                    return "white" if board.current_player_color == PlayerColor.WHITE else "black"
                elif board.current_opponent_has_won:
                    return "black" if board.current_player_color == PlayerColor.WHITE else "white"
                else:
                    return "draw"
        except Exception:
            return "draw"  # Default to draw if parsing fails
    
    def process_gamestring_file(self, input_file: str, output_file: str):
        """
        Process a file containing gamestrings and convert to training format.
        
        Args:
            input_file: Path to input file with gamestrings (one per line)
            output_file: Path to output JSON file
        """
        games_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    moves = self.parse_gamestring_to_moves(line)
                    result = self.extract_game_result(line)
                    
                    if moves:  # Only add games with actual moves
                        games_data.append({
                            "moves": moves,
                            "result": result,
                            "source_line": line_num
                        })
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(games_data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {len(games_data)} games from {input_file}")
        print(f"Saved training data to {output_file}")
    
    def analyze_dataset(self, data_file: str):
        """
        Analyze a training dataset and print statistics.
        
        Args:
            data_file: Path to JSON training data file
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            games_data = json.load(f)
        
        total_games = len(games_data)
        white_wins = sum(1 for game in games_data if game['result'] == 'white')
        black_wins = sum(1 for game in games_data if game['result'] == 'black')
        draws = sum(1 for game in games_data if game['result'] == 'draw')
        
        move_counts = [len(game['moves']) for game in games_data]
        avg_moves = sum(move_counts) / len(move_counts) if move_counts else 0
        min_moves = min(move_counts) if move_counts else 0
        max_moves = max(move_counts) if move_counts else 0
        
        print(f"Dataset Analysis for {data_file}")
        print(f"{'='*50}")
        print(f"Total games: {total_games}")
        print(f"White wins: {white_wins} ({white_wins/total_games*100:.1f}%)")
        print(f"Black wins: {black_wins} ({black_wins/total_games*100:.1f}%)")
        print(f"Draws: {draws} ({draws/total_games*100:.1f}%)")
        print(f"Average moves per game: {avg_moves:.1f}")
        print(f"Move count range: {min_moves} - {max_moves}")
        
        # Analyze vocabulary coverage
        all_moves = set()
        for game in games_data:
            all_moves.update(game['moves'])
        
        known_moves = sum(1 for move in all_moves if move in self.encoder.token_to_id)
        unknown_moves = all_moves - set(self.encoder.token_to_id.keys())
        
        print(f"Vocabulary coverage: {known_moves}/{len(all_moves)} ({known_moves/len(all_moves)*100:.1f}%)")
        if unknown_moves:
            print(f"Unknown moves (first 10): {list(unknown_moves)[:10]}")


def create_sample_training_data(output_file: str = "sample_hive_games.json"):
    """
    Create a sample training dataset with simple Hive games.
    
    Args:
        output_file: Path to output JSON file
    """
    sample_games = [
        # Basic opening sequences
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\"],
            "result": "draw"
        },
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wA1 wQ/", "bA1 bQ/"],
            "result": "white"
        },
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wA1 wQ/", "bA1 bQ/", "wG1 wA1/", "bG1 bA1/"],
            "result": "black"
        },
        # Games with Beetle stacking
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wB1 wQ/", "bB1 bQ/", "wB1 bQ"],
            "result": "white"
        },
        # Games with Grasshopper jumps
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wG1 wQ/", "bG1 bQ/", "wG1 bQ"],
            "result": "white"
        },
        # Longer strategic games
        {
            "moves": [
                "wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wA1 wQ/", "bA1 bQ/",
                "wA2 wA1/", "bA2 bA1/", "wS2 wA2/", "bS2 bA2/", "wG1 wS2/", "bG1 bS2/"
            ],
            "result": "draw"
        },
        # Quick wins
        {
            "moves": ["wS1", "bS1 wS1/", "wQ wS1-", "bQ bS1\\", "wA1 wQ/", "bA1 bQ/", "wA2 wA1/", "bA2 bA1/", "wA3 wA2/"],
            "result": "white"
        }
    ]
    
    # Replicate and vary games
    training_data = []
    
    # Add original games multiple times with slight variations
    for base_game in sample_games:
        for i in range(50):  # 50 copies of each base game
            game_copy = base_game.copy()
            training_data.append(game_copy)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample training data with {len(training_data)} games")
    print(f"Saved to {output_file}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HiveGPT data processing utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process gamestrings command
    process_parser = subparsers.add_parser('process', help='Process gamestring file to training data')
    process_parser.add_argument('input_file', help='Input file with gamestrings')
    process_parser.add_argument('output_file', help='Output JSON file')
    
    # Analyze dataset command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze training dataset')
    analyze_parser.add_argument('data_file', help='JSON training data file')
    
    # Create sample data command
    sample_parser = subparsers.add_parser('sample', help='Create sample training data')
    sample_parser.add_argument('--output', default='sample_hive_games.json', 
                              help='Output file for sample data')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        processor = HiveGameDataProcessor()
        processor.process_gamestring_file(args.input_file, args.output_file)
    elif args.command == 'analyze':
        processor = HiveGameDataProcessor()
        processor.analyze_dataset(args.data_file)
    elif args.command == 'sample':
        create_sample_training_data(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
