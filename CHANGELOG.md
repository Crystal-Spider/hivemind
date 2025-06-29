# Change Log

All notable changes to the "hivemind" project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Fix a bug in Spider pathfinding where it could miss valid destination if the path that leads to them was already partially explored by other paths.
- Add print of the depth reached during $\alpha\text{-}\beta$ pruning.

## [v1.6.2] - 2025/06/25

- Fixed a bug for which time-bounded bestmove searches would crash when finding a winning move.

## [v1.6.1] - 2025/06/19

- Fixed a bug for which time-bounded bestmove searches would return an invalid move.

## [v1.6.0] - 2025/05/09

- Fixed internal caching of queen neighbors when undoing moves.
- Slightly improved board evaluation function.
- Fixed some documentation formatting.
- Fixed an error in the evaluation function that made minmax agent think a losing move was a winning one.
- Fixed an error when trying to change the `MaxBranchingFactor` engine option.

## [v1.5.0] - 2025/04/06

- Renamed minmax agents to negamax.
- Made checking whether a piece can move without breaking the hive faster by using Tarjan's algorithm for articulation points.
- Fixed time limit parameter in negamax agents.
- Optimized negamax agents.
- Changed board evaluation function for negamax agents (draft).
- Fixed some minor issues.
- Fixed Pillbug special moves.

## [v1.4.0] - 2025/02/14

- Drastically improved board evaluation for minmax agents.
- Added `MaxBranchingFactor` engine option, to limit the amount of moves a minmax agent can expand from a single node.
- Added a print of some metrics when searching for the best move with minmax agents.
- Fix best move search with minmax agents when undos are performed.
- Fix best move search with minmax agents when new games are started.
- Implement workflows for release and testing.

## [v1.3.0] - 2025/02/13

- Drastically improve minmax agents.
- Major code changes:
  * Implemented Zobrist Hash for board states.
  * Moved almost all files.
- Minor refractors or improvements.
- Removed executable and pre-built docs from source code. They will be automatically generated for releases in future versions.

## [v1.2.0] - 2024/11/29

- Add Minmax with $\alpha\text{-}\beta$ pruning agent.
- Minor internal code changes, including minor improvements to Board and its move cache.
- Added 3 new engine options:
  * `StrategyWhite`  
    Which AI agent to use when the engine plays white.
  * `StrategyBlack`  
    Which AI agent to use when the engine plays black.
  * `NumThreads`  
    Currently does nothing, might be implemented/removed in the future.

## [v1.1.0] - 2024/11/25

- Fixes for Ladybug and Soldier Ant moves.
- Addition of random playing agent.
- Minor internal code changes.
- Addition of CHANGELOG.md.

## [v1.0.0] - 2024/11/24

- First release.
- Fully functional game engine.
- Documentation (README.md and with Sphinx).
- Prebuilt `.exe` file.

[Unreleased]: https://github.com/crystal-spider/hivemind
[README]: https://github.com/crystal-spider/hivemind#readme

[v1.6.2]: https://github.com/crystal-spider/hivemind/releases?q=v1.6.2
[v1.6.1]: https://github.com/crystal-spider/hivemind/releases?q=v1.6.1
[v1.6.0]: https://github.com/crystal-spider/hivemind/releases?q=v1.6.0
[v1.5.0]: https://github.com/crystal-spider/hivemind/releases?q=v1.5.0
[v1.4.0]: https://github.com/crystal-spider/hivemind/releases?q=v1.4.0
[v1.3.0]: https://github.com/crystal-spider/hivemind/releases?q=v1.3.0
[v1.2.0]: https://github.com/crystal-spider/hivemind/releases?q=v1.2.0
[v1.1.0]: https://github.com/crystal-spider/hivemind/releases?q=v1.1.0
[v1.0.0]: https://github.com/crystal-spider/hivemind/releases?q=v1.0.0
