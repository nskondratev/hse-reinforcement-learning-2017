# Neural Networks Reinforcement Learning
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

### Run
#### Training mode
```bash
$ python3 main.py --mode train
```
#### Test mode
```bash
$ python3 main.py --mode test
```

### Visualize log file
Example command:
```bash
$ python3 visualize_log.py --figsize 30 30 --output "train_2017-10-01.jpg" "runs/2017-10-01/dqn_Tennis-v0_log.json"
```
