# Air Hockey Simulator

This project is a simulated air hockey gaming environment. 

Either one can use this to play for fun, capture state, or train reinforcement learning models.

Currently, this project supports 6 types of reinforcement learning techniques: tabular Q-learning, Double DQN (DDQN), c51-DDQN, Dueling DDQN, Synchronous Actor/Critic (A2C), and Proximal Policy Optimization (PPO).


## Installation

This project is Python 3.6+ compatible and uses Pipenv for depencency management.

To install `pipenv`, run: `pip install pipenv`.

To install all depencencies for this project, run `pipenv install`.

To enter into the virtual environment created by `pipenv`, run `pipenv shell`.

This project uses Redis. Either pull down a Docker image or download Redis locally.

## Setup

For reference, the left mallet is the robot, and the right mallet is the opponent.

We can either train our agents with the `game.py` script. There a few cli flags we use. You have the `--model` flag which specify which reinforcement strategy we want to use. If we set `--human`, then the training script will look to Redis for the human player's input instead of tracking it's position in the air hockey environment instance else it trains against a heuristic-based approach. To have a human opponent, you need to use the web ui found [here](https://github.com/Positronic-IO/air-hockey-web-ui).

To play the game, use the `--play` flag and training will not occur.

To view and interact with the agents, you can run the web ui found in this [repo](https://github.com/Positronic-IO/air-hockey-web-ui).

## Warnings
+ Beware of how you set your rewards because these settings drastically effect the exploitation/exploration tradeoff. 

## Author
[Tony Hammack](www.tonyhammack.com)

## References

Reinforment learning information:
+ [https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)
+ [https://keon.io/deep-q-learning/](https://keon.io/deep-q-learning/)
