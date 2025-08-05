# Hivemind

## Description

A [UHP](https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol)-compliant [Hive](https://en.wikipedia.org/wiki/Hive_(game)) game engine written in Python.  
The game engine logic is loosely inspired by the [Mzinga Engine](https://github.com/jonthysell/Mzinga).

The engine comes with different AI agent configurations. More on this [below](https://github.com/Crystal-Spider/hivemind?tab=readme-ov-file#ai).

This projects also provides:  
🔹 [Releases](https://github.com/Crystal-Spider/hivemind/releases) - Prebuilt executables for Linux and Windows.  
🔹 [Documentation](https://crystal-spider.github.io/hivemind/) - Detailed codebase reference.

## Documentation

The source code is fully documented with Docstrings in [reST](https://docutils.sourceforge.io/rst.html).  
Documentation for the latest release is already live at [crystal-spider.github.io/hivemind](https://crystal-spider.github.io/hivemind/).  

The structured documentation can be generated with [Sphinx](https://www.sphinx-doc.org/en/master/).  
To build the documentation yourself, simply run the following command under the `docs/` directory:
```powershell
make html
```
To view it, simply open the file `docs/build/html/index.html` with a browser.

## Setup

Setting up the environment is pretty easy:

1. Set up **Python 3.12.7** (you can use any environment manager or none).
2. Install the dependencies from the file [`requirements.txt`](/requirements.txt).

The suggested IDE is [Visual Studio Code](https://code.visualstudio.com/), and settings for it are included.

## Usage

There are two ways to use this Hive engine:

1. Run [`engine.py`](/src/engine.py) from the command line or with your IDE and use the console to interact with it.  
   The engine will be fully functional, but there won't be any graphical interface.
2. Use the [released executables](https://github.com/Crystal-Spider/hivemind/releases) (or build one yourself) along with [MzingaViewer](https://github.com/jonthysell/Mzinga/wiki/MzingaViewer).  
   To do this, move the `HivemindEngine` executable into the same directory as `MzingaViewer` and then follow the instructions [here](https://github.com/jonthysell/Mzinga/wiki/BuildingAnEngine), specifically `step 2 > iii`.

To build the `HivemindEngine` executable yourself, simply run the following command in the project root:
```powershell
pyinstaller ./src/engine.py --name HivemindEngine --noconsole --onefile
```
This will create an executable for your platform.

## AI

There are currently 3 implemented AI strategies:

1. **Random**: the agent plays random moves.
2. **Negamax** (formerly Minmax): the agent plays moves following a Negamax policy with alpha-beta pruning and a custom node (game state) evaluation.
3. **GPT**: the agent uses a fine-tuned GPT model to predict moves and evaluate positions, similar to the ALLIE approach for chess but adapted for Hive.

### GPT Brain

The GPT brain is a neural network-based AI that learns Hive strategy through fine-tuning on game data. Key features:

- **Architecture**: Decoder-only Transformer based on GPT-2 medium (355M parameters)
- **Training**: Joint learning of move policy and position evaluation
- **Input**: Game history as token sequences
- **Output**: Move probabilities and position values

To use the GPT brain:

1. Install additional dependencies: `pip install torch transformers numpy datasets accelerate`
2. Train on game data: `python train_gpt.py --use-sample-data`
3. Use in engine: `from ai.gpt_brain import GPTBrain`

See `GPT_BRAIN_README.md` for detailed documentation.

A third machine learning implementation was planned and has now been realized with the GPT brain.
