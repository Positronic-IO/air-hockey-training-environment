# Air Hockey Simulator

This project is a simulated air hockey gaming environment. 

Either one can use this to play for fun, capture state, or train reinforcement learning models.

Currently, this project supports 5 types of reinforcement learning techniques: Q-learning, Deep Q-learning (DQN), and Double DQN (DDQN), c51-DDQN, and Dueling DDQN.

Many examples of reinforcement learning capture the state from video frames; thus, their architecture involves CNNs. This project captures the state of the board via the coordinates of agent and the puck.

Our model uses the [Huber Loss](https://en.wikipedia.org/wiki/Huber_loss) as the loss function for training. For reinforcement learning, this loss function is recommended.


## Installation

This project is Python 3.6+ compatible and uses Pipenv for depencency management.

To install `pipenv`, run: `pip install pipenv`.

To install all depencencies for this project, run `pipenv install`.

To enter into the virtual environment created by `pipenv`, run `pipenv shell`.

This project uses Redis. Either pull down a Docker image or download Redis locally.

## Setup

For reference, the left mallet is the robot, and the right mallet is the opponent.

All the settings for the different reinforcement learning strategies can be found in `configs/`. 

We can either train our agents  with the `train.py` script. There a few cli flags we use. You have `--robot` and `--opponent` flags which specify which reinforcement strategy we want to use. If we set `--opponent` to  `human`, then the training script will look to Redis for the human player's input instead of tracking it's position in the air hockey environment instance. For training, we set the "train" key in `configs/{strategy}.json` to `true`. This key is important because it will tell the algorithm to just use the model's predictions and no extra randomness. Also, the respective load and save paths for the models can be defined there. Q-learning is an exception because it uses the traditional Q-learning algorithm. There is no actual machine learning going on. Thus, there is no need to load/save the model. A save path is required for all models while a load path does not have to be defined. There is a `--capacity` cli flag which defines how many frames we want to use for training and prediction.

If we want to have our model play, we can set the train field to false and run `predict.py` which takes in the same cli flags as `train.py`, but only focuses on model predicting.

If you want to edit the rewards of the game, the rewards dictionary can be found in `environment/AirHockey.py`. You can also edit the environment and define custom reward functions.


The 	`gui.py` brings up a `pygame` gui windown, and the puck, robot's and opponent's mallets are tracked via Redis. If we have `--human` entered, then it will let the gui know to let the opponent use their mouse to move the opponent's mallet. This data is sent to Redis. You can set the fps of `gui.py` with the `--fps` flag. This defaults to `-1` which turns off the fps setting for `pygame`.

The neural network architectures for these learning strategies can be found in the `rl/Networks.py` file.

## Warnings
+ Beware of how you set your rewards because these settings drastically effect the exploitation/exploration tradeoff. 

## Todo
+ Re-enable Tesnsorboard logging
+ Have the compute and display interesting statistics of the game.
+ Explore policy based reinforcement learning strategies.

## Author
[Tony Hammack](www.tonyhammack.com)

## References

Reinforment learning information:
+ [https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)
+ [https://keon.io/deep-q-learning/](https://keon.io/deep-q-learning/)

The guts of the air hockey gui:
[https://github.com/edwardyu236/airHockey](https://github.com/edwardyu236/airHockey)