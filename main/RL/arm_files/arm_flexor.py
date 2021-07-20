import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from .osim import OsimEnv
import csv
import sys

class Arm2DEnv(OsimEnv):
    model_path = os.path.join(os.path.dirname(__file__), '14muscles_flexor.osim')
    time_limit = 200
    print("Arm Environment")

    def __init__(self, *args, **kwargs):
        super(Arm2DEnv, self).__init__(*args, **kwargs)

        self.noutput = self.osim_model.noutput

        self.osim_model.model.initSystem()

        self.target_shoulder = np.load(os.path.join(os.path.dirname(__file__), 'shoulder_zero_0_50steps_withEnding.npy'))
        self.target_elbow = np.load(os.path.join(os.path.dirname(__file__), 'elbow_plus_158_50steps_withEnding.npy'))

        self.iteration = 0

    def reset(self, random_target=False, obs_as_dict=True):
        obs = super(Arm2DEnv, self).reset(obs_as_dict=obs_as_dict)
        #if random_target:
        #    self.generate_new_target()
        state = self.osim_model.get_state()
        self.osim_model.set_state(state)
        self.osim_model.model.equilibrateMuscles(self.osim_model.state)
        self.osim_model.reset_manager()
        self.iteration = 0
        return obs

    def get_observation(self):
        state_desc = self.get_state_desc()
        res = []
        for joint in ["r_shoulder","elbow"]:
            #print(joint)
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            # res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            #res += [np.cos(state_desc["muscles"][muscle]["fiber_length"])]
            #res += [np.sin(state_desc["muscles"][muscle]["fiber_length"])]
            #res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        #res += state_desc["markers"]["r_radius_styloid"]["pos"][:2]

        return res[6:20]

    def get_positions(self):
        state_desc = self.get_state_desc()
        return state_desc["joint_pos"]["r_shoulder"][0], state_desc["joint_pos"]["elbow"][0]

    def get_observation_space_size(self):
        return 14

    def reward(self):
        state_desc = self.get_state_desc()

        step_reward_shoulder = self.target_shoulder[self.iteration]
        step_reward_elbow = self.target_elbow[self.iteration]
        penalty = (state_desc["joint_pos"]["r_shoulder"][0] - step_reward_shoulder)**2 + (state_desc["joint_pos"]["elbow"][0]-step_reward_elbow)**2
        self.iteration += 1
        return -penalty

    def get_reward(self):
        return self.reward()


class Arm2DVecEnv(Arm2DEnv):
    def reset(self, obs_as_dict=False):
        obs = super(Arm2DVecEnv, self).reset(obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
        return obs
    def step(self, action, obs_as_dict=False):
        if np.isnan(action).any():
            action = np.nan_to_num(action)
        obs, reward, done, info = super(Arm2DVecEnv, self).step(action, obs_as_dict=obs_as_dict)
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            done = True
            reward -10
        return obs, reward, done, info
