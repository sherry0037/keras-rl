"""
Train dqn model on Atari games and save RGB screenshots and RAM during training.

Run `python examples/new_dqn_atari.py --transfer` to run tests on RGB environment use pretrained RAM model

Run `python examples/new_dqn_atari.py --finetune` to use pretrained RAM model then fine-tune on RGB environment.
"""

from __future__ import division
import argparse
import os

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.agents.new_dqn import NewDQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from rgb2ram.models import *

INPUT_SHAPE = (84, 84)
RAM_SHAPE = (128,)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def __init__(self, is_ram, transfer_model=None, transfer_model_name="cnn"):
        self.is_ram = is_ram
        self.transfer_model = transfer_model
        self.transfer_model_name = transfer_model_name
        super().__init__()

    def process_observation(self, observation):
        if self.is_ram and self.transfer_model is None:
            assert observation.ndim == 1
            assert observation.shape == RAM_SHAPE
            return observation.astype('uint8') # saves storage in experience memory

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
        if self.transfer_model is not None:
            if self.transfer_model_name == "cnn":
                # processed_batch.shape = (1, 4, 84, 84)
                processed_batch = processed_batch.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)  # (4, 84, 84, 1)
                processed_batch = self.transfer_model.predict(processed_batch)
                processed_batch = processed_batch.reshape(1, processed_batch.shape[0], processed_batch.shape[1])  # (1, 4, 128)
                #processed_batch[processed_batch<0] = 0
            elif self.transfer_model_name == "ff":
                processed_batch = processed_batch.reshape(-1, INPUT_SHAPE[0]*INPUT_SHAPE[1])  # (4, 84, 84, 1)
                processed_batch = self.transfer_model.predict(processed_batch)
                processed_batch = processed_batch.reshape(1, processed_batch.shape[0],
                                                          processed_batch.shape[1])  # (1, 4, 128)
                processed_batch[processed_batch < 0] = 0
            elif self.transfer_model_name == "lstm":
                processed_batch = processed_batch.reshape(1, -1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)  # (1, 4, 84, 84, 1)
                #processed_batch = processed_batch[:, 1:, :, :, :] # (1, 3, 84, 84, 1)
                processed_batch = self.transfer_model.predict(processed_batch)
                #processed_batch = processed_batch.reshape(1, processed_batch.shape[0], processed_batch.shape[1])  # (1, 4, 128)
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


class NeuralNetworkModelBuilder():
    def __init__(self, model_name, nb_actions):
        self.model_name = model_name
        self.nb_actions = nb_actions

    def build(self):
        model = Sequential()
        if self.model_name == "rgb":
            input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
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
            model.add(Dense(self.nb_actions))
            model.add(Activation('linear'))

        elif self.model_name == "just_ram":
            model.add(Dense(128, input_shape=(4, 128), kernel_initializer='normal', activation='relu'))
            model.add(Dense(128, activation='relu', input_dim=128))
            model.add(Flatten())
            model.add(Dense(self.nb_actions))
            model.add(Activation('linear'))

        elif self.model_name == "big_ram":
            model.add(Dense(128, input_shape=(4, 128), kernel_initializer='normal', activation='relu'))
            model.add(Dense(128, activation='relu', input_dim=128))
            model.add(Dense(128, activation='relu', input_dim=128))
            model.add(Dense(128, activation='relu', input_dim=128))
            model.add(Flatten())
            model.add(Dense(self.nb_actions))
            model.add(Activation('linear'))

        return model


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
# Env-name: RAM: Breakout-ramDeterministic-v4, RGB: BreakoutDeterministic-v4
parser.add_argument('--env-name', type=str, default='Breakout-ramDeterministic-v4')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--model', choices=['rgb', 'just_ram', 'big_ram'], default="just_ram",
                    help="Choose the network from rgb|just_ram|big_ram")
parser.add_argument('--transfer_model', choices=['cnn', 'ff', 'lstm'], default="cnn",
                    help="Choose the RGB2RAM model")
parser.add_argument('--save_observations', action="store_true", default=False)
parser.add_argument('--transfer', action="store_true", default=False,
                    help="Set true to use transfer learning.")
parser.add_argument('--finetune', action="store_true", default=False,
                    help="Set true to use transfer learning then fine tune.")
parser.add_argument('--steps', type=int, default=1750000)

args = parser.parse_args()

if args.transfer:
    args.mode = 'test'
    #args.env_name = 'BreakoutDeterministic-v4'
    args.env_name = 'Breakout-v4'
    args.model = 'big_ram'
    #args.model = 'just_ram'

if args.finetune:
    args.mode = 'train'
    #args.env_name = 'BreakoutDeterministic-v4'
    args.env_name = 'Breakout-v4'
    args.model = 'big_ram'

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
model_builder = NeuralNetworkModelBuilder(args.model, nb_actions)
model = model_builder.build()
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

if args.transfer or args.finetune:
    if args.transfer_model == "cnn":
        model_type = CNNModel2
        layer_sizes = [0, 0]
        transfer_model = model_type(layer_sizes=layer_sizes, model_type=model_type).build()
        transfer_model.load_weights("./rgb2ram/saved_model/CNNModel2.h5")
    elif args.transfer_model == "ff":
        model_type = FFModel
        transfer_model = model_type(model_type=model_type).build()
        transfer_model.load_weights("./rgb2ram/saved_model/FFModel.h5")
    elif args.transfer_model == "lstm":
        model_type = LSTMModel
        transfer_model = model_type(model_type=model_type).build()
        transfer_model.load_weights("./rgb2ram/saved_model/LSTMModel.h5")
    print("RGB2RAM Model:")
    print(transfer_model.summary())
    processor = AtariProcessor(is_ram=True, transfer_model=transfer_model,
                               transfer_model_name=args.transfer_model)
else:
    processor = AtariProcessor(is_ram="ram" in args.env_name)

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

if args.save_observations:
    dqn = NewDQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                      processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                      train_interval=4, delta_clip=1.)
else:
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    save_dir = "./saved_model/" + args.model
    if args.finetune:
        save_dir = "./saved_model/" + args.model + "_finetune"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(os.path.join(save_dir, checkpoint_weights_filename), interval=250000)]
    callbacks += [FileLogger(os.path.join(save_dir, log_filename), interval=100)]

    if args.finetune:
        #pretrained_weights_filename = 'dqn_Breakout-ramDeterministic-v4_weights_1750000.h5f'
        pretrained_weights_filename = 'dqn_Breakout-ram-v4_weights_1750000.h5f'
        print("Load pre-trained weights {}.".format(pretrained_weights_filename))
        dqn.load_weights(os.path.join("./saved_model/" + args.model, pretrained_weights_filename))

    # Use new_fit to save both RGB and RAM during training
    # Use fit to train normally
    if args.save_observations:
        dqn.new_fit(env, callbacks=callbacks, nb_steps=args.steps, log_interval=10000, verbose=2)
    else:
        dqn.fit(env, callbacks=callbacks, nb_steps=args.steps, log_interval=10000, verbose=2)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=20, visualize=False)
elif args.mode == 'test':
    save_dir = "./saved_model/" + args.model
    weights_filename = 'dqn_{}_weights_{}.h5f'.format(args.env_name, args.steps)
    if args.transfer:
        #weights_filename = 'dqn_Breakout-ramDeterministic-v4_weights_{}.h5f'.format(args.steps)
        weights_filename = 'dqn_Breakout-ram-v4_weights_{}.h5f'.format(args.steps)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(os.path.join(save_dir, weights_filename))
    dqn.test(env, nb_episodes=20, visualize=False)
