## Yahtzee MDP

Python of the game of Yahtzee as a Markov Decision Process and different algorithms (Value Iteration, MCTS) to play the game.

## Dependencies

Python 3.10 and numpy 1.26.4 (can be implemented without any dependencies but the engine would be slower)

## Content

- engine/
  - game_engine.py :  Meta class for game engines (should be refactored into a class for Markov Decision Process)
  - yahtzee_engine.py : Yahtzee implementation of GameEngine class with caching and lazy evaluation of transitions.
  - constants.py : File with constant variables
- mcts.py : Implementation of the Monte Carlo Search Tree algorithm with UCT expansion
- multi_mcts.py : Implementation of root parallelized MCTS algorithm
- value_iteration.py : Implementation of the Value Iteration algorithm

## TODO

- Cascade last changes of mcts.py into multi_mcts.py

## Further improvments

- Add remaining upper score point before bonus as a state features + re-evaluate performance
- Try some Q-function approximation techniques
- Try policy-based approaches
- Implementation in low-level language to improve performance
- Refactor the GameEngine meta class into Markov Decision Process meta class

## Meta

Yacine BEKKA – [Linkedin](https://www.linkedin.com/in/yacine-bekka-519b79146) – yacinebekka@yahoo.fr

Distributed under MIT license. See ``LICENSE`` for more information.

