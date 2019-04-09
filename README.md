# Air Hockey Simulator

This project is a simulated air hockey gaming environment. 

Either one can use this to play for fun, capture state, or train reinforcement learning models.

Currently, this project supports 3 types of reinforcement learning techniques: Q-learning, Deep Q-learning (DQN), and Double DQN (DDQN).

Many examples use of reinforcement learning capture the state from video frames; thus, their architecture involves CNNs. This project captures the state of the board via the coordinates of agent, puck, and opponent. The main neural net archcitecture is a multilayer percecptron.

Our model uses the [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) as the loss function for training. For reinforcement learning, this loss function is recommended.


## Installation

This project is Python 3.6+ compatible and uses Pipenv for depencency management.

To install `pipenv`, run: `pip install pipenv`.

To install depencencies for this project, run `pipenv install`.

To enter into the virtual environment created by `pipenv`, run ``pipenv shell`.

## Setup

There are two ways to use this simulator: either with a human agent or AI agent. These can set via command line flags. If you wish to use an AI agent, you can specify where to either load a previously trained model or where to save a model. Since reinforcement learning updates weights iteratively for predictions, we must specify a path to save a model. These paths can be set via command line flags as well. As a default, the model is saved periodically and denoted with a number for versioning. (If you loaded a model, you do not necessarily have to set a save path. It will use the path for loading as its path for saving if a path is not specified.)

Q-learning is an exception because it uses the traditional Q-learning algorithm (There is no actual machine learning going on.)  Thus, there is no need to load/save the model.

## Run Simulator

In our virtual environment, we can start the simulator with a human agent via `python3 start.py --agent human`. Now, you can use your mouse to as a controller for the puck.

If you want a robot agent, run `python3 start.py --agent robot`. This defaults to Q-learning. To set the learning strategy, run 

```
python3 start.py --agent robot --strategy dqn-learner
python3 start.py --agent robot --strategy ddqn-learner
```

for either the DQN or DDQN learning strategies.

To save/load a model, run `python3 start.py --agent robot --strategy <your strategy> --load <load path> --save <save path>`.

You can also run this simulation in "test mode" with the flag `--env test.` The test environment only allows the puck to be played in the agent's half of the board. Also, all scoring is off. This is a good way to prepare your model for game training. Training in this simulated environment allows the model to learn basic puck interactions due to these interactions happening more frequently.

If you want to run the simulation without a gui (or headless), add `--mode cli`. All the command line configuration is the same. Obviously, your agent has to be the robot because there is no gui.

To define the fps of the gui, enter the desired fps after the `--fps` flag. The default fps is 60. A value of -1 turns off the fps throttling. FPS does not affect `cli` mode.

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