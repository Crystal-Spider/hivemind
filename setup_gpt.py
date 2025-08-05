#!/usr/bin/env python3
"""
Setup and installation script for HiveGPT brain.
"""

import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch',
        'transformers', 
        'numpy',
        'datasets',
        'accelerate'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    return missing_packages

def install_dependencies(packages):
    """Install missing packages."""
    if not packages:
        print("All dependencies are already installed!")
        return True
    
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    
    # Determine pip install commands based on packages
    if 'torch' in packages:
        # Install PyTorch with appropriate version
        print("Installing PyTorch...")
        cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu']
        try:
            subprocess.run(cmd, check=True)
            packages.remove('torch')
        except subprocess.CalledProcessError as e:
            print(f"Error installing PyTorch: {e}")
            return False
    
    # Install remaining packages
    if packages:
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        try:
            subprocess.run(cmd, check=True)
            print("✓ All packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            return False
    
    return True

def test_installation():
    """Test if the GPT brain can be imported and used."""
    try:
        from ai.gpt_brain import GPTBrain, HiveGameEncoder
        from core.board import Board
        
        print("\nTesting GPT brain installation...")
        
        # Test encoder
        encoder = HiveGameEncoder()
        print(f"✓ Encoder created with {encoder.vocab_size} vocabulary size")
        
        # Test model creation (without loading weights)
        brain = GPTBrain()
        print("✓ GPT brain initialized")
        
        # Test basic functionality
        board = Board()
        move = brain.find_best_move(board, max_branching_factor=5)
        print(f"✓ Move prediction works: {move}")
        
        evaluation = brain.evaluate_position(board)
        print(f"✓ Position evaluation works: {evaluation:.3f}")
        
        print("\n🎉 Installation successful! GPT brain is ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("HiveGPT Brain Setup")
    print("=" * 40)
    
    # Check current directory
    if not Path("src/ai/gpt_brain.py").exists():
        print("❌ Error: Please run this script from the hivemind project root directory")
        sys.exit(1)
    
    # Add src to Python path for testing
    sys.path.insert(0, str(Path.cwd() / "src"))
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        response = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if install_dependencies(missing):
                print("\n✓ Dependencies installed successfully!")
            else:
                print("\n❌ Failed to install dependencies")
                sys.exit(1)
        else:
            print("Setup cancelled. Install dependencies manually:")
            print("pip install torch transformers numpy datasets accelerate")
            sys.exit(1)
    
    # Test installation
    if test_installation():
        print("\nNext steps:")
        print("1. Create training data: python data_utils.py sample")
        print("2. Train model: python train_gpt.py --use-sample-data")
        print("3. Test usage: python example_gpt_usage.py")
    else:
        print("\nSetup completed but testing failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
