# Air Hockey Simulator

This project is a simulated air hockey gaming environment. 

Either one can use this to play for fun, capture state, or train reinforcement learning models.

Currently, this project supports 3 types of reinforcement learning techniques: Q-learning, Deep Q-learning (DQN), and Double DQN (DDQN).

Many examples use of reinforcement learning capture the state from video frames; thus, their architecture involves CNNs. This project captures the state of the board via the coordinates of agent, puck, and opponent.

Ours model uses the [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) as the loss function for training. For reinforcement learning, this loss function is recommended.


## Installation

This project is Python 3.6+ compatible and uses Pipenv for depencency management.

To install `pipenv`, run: `pip install pipenv`.

To install depencencies for this project, run `pipenv install`.

To enter into the virtual environment created by `pipenv`, run ``pipenv shell`.

This project uses Redis. Either pull down a Docker image or download Redis locally.

## Setup

There are two ways to use this simulator: either with a human agent or AI agent. These can set via command line flags. If you wish to use an AI agent, you can specify where to either load a previously trained model or where to save a model. Since reinforcement learning updates weights iteratively for predictions, we must specify a path to save a model. These paths can be set via command line flags as well. As a default, the model is saved periodically and denoted with a number for versioning. (If you loaded a model, you do not necessarily have to set a save path. It will use the path for loading as its path for saving if a path is not specified.)

Q-learning is an exception because it uses the traditional Q-learning algorithm (There is no actual machine learning going on.)  Thus, there is no need to load/save the model.

## Run Simulator

There are two important scripts in the repo, `gui.py` and `main.py`. `gui.py` brings up a gui of the air hockey environment and either allows the user to play with their mouse or display results from a robot agent vis Redis updates. `main.py` controls what type of learning strategy your robot wants to use.

In our virtual environment, we can start the gui with either a human or robot agent eith a specific fps via `python3 gui.py --agent <agent> --fps <fps>`. If a robot agent is chosen, there is a note to start the robot learning strategy via `main.py`.

If you want a robot agent, run `python3 main.py --strategy <your strategy> --load <load path> --save <save path>` to set up its learning strategy.

If you want to throttle the speed of the robot, there is a flag `--wait` for `main.py`. The units for `--wait` are in seconds.

## Warnings
+ Beware of how you set your rewards because these settings drastically effect the exploitation/exploration tradeoff. 
+ In the DQN and DDQN model achitecture, there is no dropout or any type of normalization. There is no need for this in reinforcement learning. Overfitting can be a good thing. If you feel your model is not robust enough, try introducing more noise into the system via modifying the gameplay.

## Todo
+ Have the agent duel against itself.

## Miscellaneous
This project uses Python type hints. You can use [Mypy](https://mypy.readthedocs.io/en/latest/) or any other static type checking system.

## Author
[Tony Hammack](www.tonyhammack.com)

## References

Reinforment learning information:
[https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)
[https://keon.io/deep-q-learning/](https://keon.io/deep-q-learning/)

The guts of the air hockey gui:
[https://github.com/edwardyu236/airHockey](https://github.com/edwardyu236/airHockey)