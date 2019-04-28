# Air Hockey Simulator

This project is a simulated air hockey gaming environment. 

Either one can use this to play for fun, capture state, or train reinforcement learning models.

Currently, this project supports 5 types of reinforcement learning techniques: Q-learning, Deep Q-learning (DQN), and Double DQN (DDQN), c51-DDQN, and Dueling DDQN.

Many examples use of reinforcement learning capture the state from video frames; thus, their architecture involves CNNs. This project captures the state of the board via the coordinates of agent, puck, and opponent.

Our model uses the [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) as the loss function for training. For reinforcement learning, this loss function is recommended.


## Installation

This project is Python 3.6+ compatible and uses Pipenv for depencency management.

To install `pipenv`, run: `pip install pipenv`.

To install depencencies for this project, run `pipenv install`.

To enter into the virtual environment created by `pipenv`, run ``pipenv shell`.

This project uses Redis. Either pull down a Docker image or download Redis locally.

## Setup

For reference, the left mallet is the main agent, and the right mallet is the opponent.

All the settings can be found in JSON config files. The config files for the reinforcement learning strategies (these detail the hyperparameters) can be found in the `rl/configs` directory. The main config file is `config.json`.

We can either train our agents or play air hockey. This can be set with the "train" flag in `config.json`. If we are training, then the training script will look at the "training" field in `config.json` for the strategies we ought to use. Also, the respective load and save paths for the models can be defined there. Q-learning is an exception because it uses the traditional Q-learning algorithm. There is no actual machine learning going on. Thus, there is no need to load/save the model. A save path is required for all models while a load path does not have to be defined. We can also define a path to save stats about the training in a csv file by the "results" subfield of the "training" field.

If we want to have our model play, we can set the train field to false and the simulator will use the strategies defined in the `live` field. We can define where we should load the models.

In the main config file, we can define the rewards we want to use to train the strategies. Currently, it is set to train the strategies based of whether an agent scores or loses. You can edit the game environment to include more complex rewards.

There is a field called "capcity" which defines how many frames we want to use for training.

You can set the fps of `gui.py` with the "fps" field. This defaults to `-1` which turns off the fps setting for `pygame`.

The "tensorboard" field details where you want to save Tensorboard logs.

The "robot" field dictates whether we want the main agent (the left mallet) to be either a human or a robot agent.

## Run Simulator

There are two important scripts in the repo, `gui.py` and `train.py`. `gui.py` brings up a gui of the air hockey environment and either allows the user to play with their mouse, a loaded model, or display results from a robot agent training via Redis updates. `train.py` controls what type of learning strategy your robot wants to use.

In our virtual environment, we can start the gui with either a human or robot agent with a specific fps via `python3 gui.py` and have the appropriate fields in the `config.json` file defined. If you want to load a model, you also have to specify the strategy your agent wants to use. (This is due to each strategy having certain hyperparameters.)

If you want a robot agent, run `python3 train.py` and have the appropriate fields in the `config.json` file defined to set up its learning strategy and duel against another agent.

The neural network architectures for these learning strategies can be found in the `rl/Networks.py` file.

## Warnings
+ Beware of how you set your rewards because these settings drastically effect the exploitation/exploration tradeoff. 

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