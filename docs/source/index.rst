Hivemind documentation
======================

Description
-----------

| A `UHP <https://github.com/jonthysell/Mzinga/wiki/UniversalHiveProtocol>`_-compliant `Hive <https://en.wikipedia.org/wiki/Hive_(game)>`_ game engine written in Python.  
| The game engine logic is loosely inspired from `Mzinga Engine <https://github.com/jonthysell/Mzinga>`_.
|
| The engine comes with different AI agent configurations. More on this below.
|
| This projects also provides:
| 🔹 `Releases <https://github.com/Crystal-Spider/hivemind/releases>`_ - Prebuilt executables for Linux and Windows.  
| 🔹 `Documentation <https://crystal-spider.github.io/hivemind/>`_ - Detailed codebase reference.

Setup
-----

Setting up the environment is pretty easy:

1. Set up **Python 3.12.7** (you can use any environment manager or none).
2. Install the dependencies from the file ``requirements.txt``.

The suggested IDE is `Visual Studio Code <https://code.visualstudio.com/>`_, and settings for it are included.

Usage
-----

There are two ways to use this Hive engine:

1. | Run ``engine.py`` from the command line or with your IDE and start using the console to interact with it.
   | The engine will be fully functional, but there won't be any graphical interface.
2. | Use the `released executables <https://github.com/Crystal-Spider/hivemind/releases>`_ (or build one yourself) along with `MzingaViewer <https://github.com/jonthysell/Mzinga/wiki/MzingaViewer>`_.
   | To do this, move the ``HivemindEngine`` executable into the same directory as ``MzingaViewer`` and then follow the instructions `here <https://github.com/jonthysell/Mzinga/wiki/BuildingAnEngine>`_, specifically ``step 2 > iii``.

To build the ``HivemindEngine`` executable yourself, simply run the following command in the project root:

.. code:: powershell

   pyinstaller ./src/engine.py --name HivemindEngine --noconsole --onefile

This will create an executable for your platform.

AI
---

There are currently 3 implemented AI strategies:

1. **Random**: the agent plays random moves.
2. **Negamax** (formerly Minmax): the agent plays moves following a Negamax policy with alpha-beta pruning and a custom node (game state) evaluation.
3. **GPT**: the agent uses a fine-tuned GPT model to predict moves and evaluate positions, similar to the ALLIE approach for chess but adapted for Hive.

The GPT brain represents a machine learning approach that learns Hive strategy through fine-tuning on game data.

Contents
--------

.. toctree::
   :maxdepth: 2

   engine
   core/index
   ai/index
