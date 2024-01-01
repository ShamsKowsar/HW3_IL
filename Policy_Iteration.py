
import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

class PolicyIteration():
    def __init__(self, env, discount, theta):
        self.env = env
        self.discount = discount
        self.theta = theta
        self.num_states=env.get_state_size()
        self.num_actions=env.get_action_size()
        self.values=np.zeros(self.num_states)
        self.policy=np.zeros(self.num_states,self.num_actions)
    def calc_1(self,state):
        action=np.argmax(self.policy[state])
        direction = self._action_to_direction[action]
        action_with_wind_prob=random.uniform(0,1)
        value=0
        possible_next_states=[]
        possible_next_states.append([np.clip(self._agent_location + direction, 0, self.size - 1),0.8])
        possible_next_states.append([np.clip(self._agent_location + direction*-1, 0, self.size - 1),0.1])
        possible_next_states.append([np.clip(self._agent_location, 0, self.size - 1),0.1])
        if self.age





    def policy_evaluation(self):
        delta = np.inf

        while(delta < self.theta):

            self.delta = 0

            for state in range(self.num_states):
              temp_v=self.values[state]
              self.values[state]=calc_1(state)
              delta=max(delta,abs(temp_v-self.values[state]))

    def policy_improvement(self):
        self.policy_stable = True
        for state in range(self.num_states):
          old_action=self.policy(state)
          self.update_policy(state)
          if old_action != self.policy(state):
            self.policy_stable=False


    def fit(self):
        while(not self.policy_stable):
            self.policy_evaluation()
            self.policy_improvement()

    def get_optimal_policy(self, state):
        return np.argmax(self.q_values[state[X], state[Y], :])

    def get_state_values(self):
        return self.state_values

    def get_q_values(self):
        return self.q_values

    def reset(self):
        self.env.reset()
        self.delta = 0
        self.state_values = np.zeros((self.env.map_size, self.env.map_size))
        self.q_values = np.zeros((self.env.map_size, self.env.map_size, 4))
        self.optimal_policy = np.random.randint(4, size=(self.env.map_size, self.env.map_size))
        self.policy_stable = False
    