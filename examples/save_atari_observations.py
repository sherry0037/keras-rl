from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.new_dqn import NewDQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--game_name', default='Breakout')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--steps', type=int, default=1750000)
parser.add_argument('--save_every_episode', type=int, default=5)
parser.add_argument('--save_every_step', type=int, default=5)
args = parser.parse_args()

env_name = args.game_name + '-v4'
# if args.game_name == 'Breakout':
#     env_name = 'Breakout-v4'
# if args.game_name == 'Seaquest':
#     env_name = 'Seaquest-v4'

# Get the environment and extract the number of actions.
env = gym.make(env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_data_format() == 'channels_last':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_data_format() == 'channels_first':
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

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = NewDQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                      processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                      train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    save_dir = "./saved_model/rgb"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(os.path.join(save_dir, checkpoint_weights_filename), interval=250000)]
    callbacks += [FileLogger(os.path.join(save_dir, log_filename), interval=100)]
    dqn.new_fit(env, callbacks=callbacks, nb_steps=args.steps, log_interval=10000, verbose=2,
                save_every_episode=args.save_every_episode, save_every_step=args.save_every_step)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(os.path.join(save_dir, weights_filename), overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=20, visualize=False)
elif args.mode == 'test':
    save_dir = "./saved_model/rgb"
    weights_filename = 'dqn_{}_weights.h5f'.format(env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(os.path.join(save_dir, weights_filename))
    dqn.test(env, nb_episodes=20, visualize=True)
