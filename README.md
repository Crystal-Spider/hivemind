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

There are currently 2 implemented AI strategies:

1. Random: the agent plays random moves.
2. Minmax: the agent plays moves following a Minmax policy with alpha-beta pruning and a custom node (game state) evaluation.

A third implementation will come in the future that will leverage machine learning.
