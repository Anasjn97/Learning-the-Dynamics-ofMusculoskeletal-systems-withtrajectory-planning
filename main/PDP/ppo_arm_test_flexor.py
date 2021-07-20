#!/usr/bin/env python

"""ppo_arm_test_flexor.py: Source code of the model learning on 2D arm control project
This module demonstrates how to use a gym musculoskeletal environment to learn a model to
do uesr defined trajectory mimicking
Example:
    You can directly execute with python command ::
        $ python ppo_arm_test_flexor.py

It saves the best models every user-defined steps for comparison
Options:
  -mi       --model_id            Model ID
  -c        --counter             Number of Tests
  -ns       --num_steps           Number of Steps

Options to be included:
   -hs --hidden_size      Hidden size
Attributes:
    Continue the documentation here
Todo:
    * Complete the documentation
    * Use ``sphinx.ext.todo`` extension
"""

__author__ = "Berat Denizdurduran"
__copyright__ = "Copyright 2020, Berat Denizdurduran"
__license__ = "private, unpublished"
__version__ = "1.0.0"
__email__ = "bdenizdurduran@gmail.com"
__status__ = "Production"

import math
import random
import sys

import os
from osim.env.arm_flexor import Arm2DVecEnv, Arm2DEnv

import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import LogNormal

import matplotlib.pyplot as plt

import argparse

use_cuda = torch.cuda.is_available()
print(use_cuda)
input("cuda")
device   = torch.device("cuda" if use_cuda else "cpu")

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from multiprocessing_env import SubprocVecEnv

num_envs = 1

def make_env():
    def _thunk():
        env = Arm2DVecEnv(visualize=False)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = Arm2DVecEnv(visualize=False)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.01)
        nn.init.constant_(m.bias, 0.1)




class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.PReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1),
            #nn.LeakyReLU(negative_slope=0.2)
            #nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            #nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            #nn.Linear(hidden_size, hidden_size).double(),
            #nn.Tanh().double(),
            #nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            #nn.Sigmoid(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh(),
            nn.Threshold(0.0, 0.0)
            #nn.ReLU6()
            #nn.Dropout(p=0.2)
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std).data.squeeze()

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        std = std.to(device)
        dist  = Normal(mu, std*0.1)
        return dist, value



def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig("arm_ppo_flexor{}".format(frame_idx))
    #print("elma")
    plt.close()
    #plt.show()


def test_env(num_steps, count):
    state = env.reset()
    target_shoulder = np.load("shoulder_zero_0_50steps_withEnding.npy")
    target_elbow = np.load("elbow_plus_158_50steps_withEnding.npy")

    state_shoulder = []
    state_elbow = []
    action_muscle1 = []
    action_muscle2 = []
    #if vis: env.render()
    done = False
    total_reward = 0

    for i in range(num_steps):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model_flexor(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        #input("Video")
        positions = env.get_positions()
        state_shoulder.append(positions[0])
        state_elbow.append(positions[1])

        action_muscle1.append(action[0])
        action_muscle2.append(action[1])
        state = next_state
        total_reward += reward

    plt.plot(target_elbow)
    plt.plot(state_elbow)
    plt.plot(target_shoulder)
    plt.plot(state_shoulder)
    plt.savefig("arm_ppo_states_all_flexor_{}_{}".format(model_id, count))
    plt.close()
    plt.plot(target_shoulder)
    plt.plot(state_shoulder)
    plt.savefig("arm_ppo_states_shoulder_flexor_{}_{}".format(model_id, count))
    plt.close()
    envs.reset()
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.9, tau=0.99):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]



def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model_flexor(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer_flexor.zero_grad()
            loss.backward()
            optimizer_flexor.step()


## define the argparse items
if __name__ == "__main__":
    ## define the argparse items
    #random.seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument("-mi", "--model_id", type=int, default=300, help="Model ID")
    parser.add_argument("-c", "--counter", type=int, default=1, help="number of tests")
    parser.add_argument("-ns", "--num_steps", type=int, default=75, help="number of steps")

    args = parser.parse_args()


    num_inputs  = 14#envs.observation_space.shape[0]
    num_outputs = 14#envs.action_space.shape[0]

    state = envs.reset()

    #Hyper params:
    hidden_size      = 32
    lr               = 3e-4
    betas            = (0.9, 0.999)
    eps              = 1e-08
    weight_decay     = 0.001
    mini_batch_size  = 200
    ppo_epochs       = 200
    threshold_reward = -200

    model_flexor = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer_flexor = optim.Adam(model_flexor.parameters(), lr=lr)



    model_id = args.model_id
    counter = args.counter
    num_steps = args.num_steps

    ppo_model_arm_flexor_loaded = torch.load("results/ppo_model_arm_flexor_{}".format(model_id), map_location=device)

    model_flexor.load_state_dict(ppo_model_arm_flexor_loaded['model_state_dict'])
    optimizer_flexor.load_state_dict(ppo_model_arm_flexor_loaded['optimizer_state_dict'])


    frame_idx = ppo_model_arm_flexor_loaded['epoch']

    test_rewards = ppo_model_arm_flexor_loaded['loss']

    test_all_rewards = np.array([test_env(num_steps, i) for i in range(counter)])
