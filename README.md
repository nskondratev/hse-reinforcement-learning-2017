# Neural Networks Reinforcement Learning
## Final presentation
You can download our final presentation [here](https://github.com/nskondratev/hse-reinforcement-learning-2017/raw/master/presentations/Neural%20Networks%20Reinforcement%20Learning.pdf).
## Goal of the project
To get knowledge about Reinforcement Learning with neural networks and implement Atari Tennis AI bot with the use of RL.
## Members
* Ketkov Sergey @sketkov1994
* Kondratev Nikita @nskondratev - nskondratyev@yandex.ru
* Makarova Olga @ovmakarova
* Pribytkina Daria @dapribytkina
* Semenov Sergey @Serghsem
## Useful links
* [Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
* [Machine Learning for Humans, Part 5: Reinforcement Learning](https://medium.com/machine-learning-for-humans/reinforcement-learning-6eacf258b265)
* [5 Ways to Get Started with Reinforcement Learning](https://buzzrobot.com/5-ways-to-get-started-with-reinforcement-learning-b96d1989c575)
### Scientific articles
* [Reinforcement Learning: An Introduction](http://incompleteideas.net/sutton/book/bookdraft2017june.pdf)
* [Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.](https://github.com/nskondratev/hse-reinforcement-learning-2017/raw/master/info/mnih2015.pdf)
### Examples
* https://github.com/dennybritz/reinforcement-learning
* https://github.com/Tantun/AIVoid
* https://github.com/yukezhu/tensorflow-reinforce
* https://github.com/zsdonghao/tensorlayer
* https://github.com/zsdonghao/seq2seq-chatbot

## Project structure
The structure of our project is the following:
* **info** - the directory with the article that contains model description.
* **presentations** - the directory with our slides.
* **runs** - the directory with training logs and weights grouped by date.
* **train_results** - the directory that contains visualized training logs.
* **atari_processor.py** - script with AtariProcessor class implementation. AtariProcessor class is used for processing environment observations and rewards.
* **main.py** - main script that is used for launching training or testing. Also contains model initialization.
* **visualize_log.py** - script for visualizing training logs.

## Used environments
We are using [OpenAI Gym](https://gym.openai.com/read-only.html) environments in this project.
We provide trained models for the following environments:
* [Atari Boxing](https://gym.openai.com/envs/Boxing-v0/)
* [Atari Robotank](https://gym.openai.com/envs/Robotank-v0)
* [Atari Tennis](https://gym.openai.com/envs/Tennis-v0)

All of this environments provides an RGB image of the screen as an observation on each step.
The input image is an array of shape (210, 160, 3).

## Usage
### Prerequisites
* Linux Ubuntu (tested on 16.04.3): gym[atari] not available on Windows
* Python 3.5.2
* Pip 9.0.1
* Installed packages: keras-rl, tensorflow, numpy, h5py, gym, gym\[atari\]

To install needed packages run the following commands:
```bash
$ sudo apt-get update && sudo apt-get install python3 python3-pip
$ pip3 install keras-rl tensorflow numpy h5py gym
$ pip3 install gym[atari]
```

If you want to visualize logs, you also need to install matplotlib:
```bash
$ pip3 install matplotlib
```

We advise you to use [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), so you can have isolated python environments with their own packages.

### Run
#### Training mode
To run DQN model in training mode you should run *main.py* script and provide argument *--mode* with the *train* value.
You also need to specify environment with the *--env* argument.

Example:
```bash
$ python3 main.py --mode train --env Boxing-v0
```
#### Test mode
To run DQN model in training mode you should run *main.py* script and provide argument *--mode* with the *test* value.
You also need to specify environment with the *--env* argument.
By default, script will search for weights of trained model in *trained_models* directory, but you can provide custom weights by specifying *--weights* argument with the path to weights file.

Example:
```bash
$ python3 main.py --mode test -- env Boxing-v0
```

### Visualize log file
After training is finished you can use the generated log file to visualize training process.
To generate charts you need to run *visualize_log.py* script and specify the following arguments:
* **--figsize** - the size of each chart,
* **--output** - the path to output image,
* **log_filename** - the path to the log file.

Example:
```bash
$ python3 visualize_log.py --figsize 30 30 --output "train_2017-10-01.jpg" "runs/2017-10-01/dqn_Tennis-v0_log.json"
```

## Experiments
We used DQN algorithm implementation in [Keras-RL](http://keras-rl.readthedocs.io/en/latest/agents/dqn/) package.
Hyperparameters are described in Mnih et al. (2015).

We trained agents in three Open AI Gym Atari environments with the same hyperparameters and the same model architecture.

| Environment 	| # of steps 	|
|-------------	|------------	|
| Boxing-v0   	| 10 000 000 	|
| Robotank-v0 	| 3 000 000  	|
| Tennis-v0   	| 10 000 000 	|

### Boxing-v0
The trained agent in Boxing-v0 environment gain the following average reward value: 23.7

#### Demo:

![Boxing demo](https://raw.githubusercontent.com/nskondratev/hse-reinforcement-learning-2017/master/trained_models_demo/Boxing.gif)

### Robotank-v0
The trained agent in Robotank-v0 environment gain the following average reward value: 8.4

#### Demo:

![Robotank demo](https://github.com/nskondratev/hse-reinforcement-learning-2017/raw/master/trained_models_demo/Robotank.gif)

### Tennis-v0
The trained agent in Tennis-v0 environment, unfortunately, decided not to play tennis at all. So that it can not gain the negative reward.
Proposed solution: to train agent on 50 million steps.

#### Demo:

![Tennis demo](https://github.com/nskondratev/hse-reinforcement-learning-2017/raw/master/trained_models_demo/Tennis.gif)

## Further work
To get better results the number of iterations in training need to be increased to at least 50 million steps. Furthermore, some environments has best results with the agents, that were trained on 80 million steps.

