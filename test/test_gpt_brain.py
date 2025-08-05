#!/usr/bin/env python3
"""
Unit tests for GPT Brain implementation.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.board import Board
from core.enums import PlayerColor

# Test basic functionality without requiring PyTorch dependencies
class TestGPTBrainBasic(unittest.TestCase):
    """Basic tests that don't require PyTorch."""
    
    def test_encoder_vocabulary(self):
        """Test that the encoder builds a reasonable vocabulary."""
        try:
            from ai.gpt_brain import HiveGameEncoder
            encoder = HiveGameEncoder()
            
            # Check vocabulary size is reasonable
            self.assertGreater(encoder.vocab_size, 100)
            self.assertLess(encoder.vocab_size, 10000)
            
            # Check special tokens are included
            self.assertIn(encoder.GAME_START, encoder.token_to_id)
            self.assertIn(encoder.MOVE_SEP, encoder.token_to_id)
            self.assertIn(encoder.PLAYER_WHITE, encoder.token_to_id)
            self.assertIn(encoder.PLAYER_BLACK, encoder.token_to_id)
            
            # Check basic moves are in vocabulary
            self.assertIn("wS1", encoder.token_to_id)
            self.assertIn("bQ", encoder.token_to_id)
            self.assertIn("pass", encoder.token_to_id)
            
        except ImportError:
            self.skipTest("PyTorch dependencies not available")
    
    def test_game_encoding(self):
        """Test encoding of game sequences."""
        try:
            from ai.gpt_brain import HiveGameEncoder
            encoder = HiveGameEncoder()
            
            # Test simple game
            moves = ["wS1", "bS1 wS1/", "wQ wS1-"]
            result = "white"
            
            sequence = encoder.encode_game_sequence(moves, result)
            
            # Check sequence is not empty
            self.assertGreater(len(sequence), 0)
            
            # Check all tokens are valid
            for token_id in sequence:
                self.assertIn(token_id, encoder.id_to_token)
            
            # Check game start token is included
            start_token_id = encoder.token_to_id[encoder.GAME_START]
            self.assertIn(start_token_id, sequence)
            
        except ImportError:
            self.skipTest("PyTorch dependencies not available")
    
    def test_data_processor(self):
        """Test the game data processor."""
        try:
            from data_utils import HiveGameDataProcessor
            processor = HiveGameDataProcessor()
            
            # Test parsing simple gamestring
            gamestring = "Base;InProgress;White[3];wS1;bS1 wS1/;wQ wS1-"
            moves = processor.parse_gamestring_to_moves(gamestring)
            
            expected_moves = ["wS1", "bS1 wS1/", "wQ wS1-"]
            self.assertEqual(moves, expected_moves)
            
        except ImportError:
            self.skipTest("Dependencies not available")


class TestGPTBrainIntegration(unittest.TestCase):
    """Integration tests requiring full dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            from ai.gpt_brain import GPTBrain
            self.gpt_available = True
        except ImportError:
            self.gpt_available = False
    
    def test_gpt_brain_initialization(self):
        """Test GPT brain can be initialized."""
        if not self.gpt_available:
            self.skipTest("GPT dependencies not available")
        
        from ai.gpt_brain import GPTBrain
        
        # Should not raise exception
        brain = GPTBrain()
        self.assertIsNotNone(brain)
        self.assertIsNotNone(brain.encoder)
        self.assertIsNotNone(brain.model)
    
    def test_gpt_brain_move_prediction(self):
        """Test GPT brain can predict moves."""
        if not self.gpt_available:
            self.skipTest("GPT dependencies not available")
        
        from ai.gpt_brain import GPTBrain
        
        brain = GPTBrain()
        board = Board()
        
        # Should return a valid move string
        move = brain.find_best_move(board, max_branching_factor=5)
        self.assertIsInstance(move, str)
        self.assertGreater(len(move), 0)
    
    def test_gpt_brain_position_evaluation(self):
        """Test GPT brain can evaluate positions."""
        if not self.gpt_available:
            self.skipTest("GPT dependencies not available")
        
        from ai.gpt_brain import GPTBrain
        
        brain = GPTBrain()
        board = Board()
        
        # Should return a float in [-1, 1] range
        evaluation = brain.evaluate_position(board)
        self.assertIsInstance(evaluation, float)
        self.assertGreaterEqual(evaluation, -1.0)
        self.assertLessEqual(evaluation, 1.0)


class TestTrainingComponents(unittest.TestCase):
    """Test training-related components."""
    
    def test_dataset_creation(self):
        """Test dataset can be created from game data."""
        try:
            from ai.gpt_brain import HiveDataset, HiveGameEncoder
            
            encoder = HiveGameEncoder()
            games_data = [
                {"moves": ["wS1", "bS1 wS1/"], "result": "white"},
                {"moves": ["wS1", "bS1 wS1/", "wQ wS1-"], "result": "black"}
            ]
            
            dataset = HiveDataset(games_data, encoder, max_length=64)
            
            self.assertEqual(len(dataset), 2)
            
            # Test getting an item
            item = dataset[0]
            self.assertIn('input_ids', item)
            self.assertIn('labels', item)
            self.assertIn('values', item)
            
        except ImportError:
            self.skipTest("PyTorch dependencies not available")


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGPTBrainBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestGPTBrainIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingComponents))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
