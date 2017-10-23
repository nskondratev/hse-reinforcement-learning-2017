import argparse
import os

import gym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from atari_processor import AtariProcessor

TRAINED_MODELS_DIR = 'trained_models'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['Robotank-v0', 'Boxing-v0', 'Tennis-v0', 'Pong-v0'], default='Boxing-v0')
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Compile agent
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env)
    checkpoint_weights_filename = 'dqn_' + args.env + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=3000000, log_interval=10000)

    # Save final weights
    dqn.save_weights(weights_filename, overwrite=True)

    # Evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = os.path.join(TRAINED_MODELS_DIR, 'dqn_{}_weights.h5f'.format(args.env))
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
