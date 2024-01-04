
import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

class Q_Learning():
  def __init__(self,env,learning_rate,discount_factor,epsilon,with_decreasing_learning_rate=False) :
    self.env=env
    self.learning_rate=learning_rate
    self.discount_factor=discount_factor
    self.epsilon=epsilon
    self.initial_epsilon=epsilon
    self.num_states=env.get_state_size()
    self.num_actions=env.get_action_size()
    self.q_table=np.zeros((env.get_state_size(),env.get_action_size()))
    self.with_decreasing_learning_rate=with_decreasing_learning_rate
    
  def select_action(self,current_state):
    _=random.uniform(0,1)
    if _<self.epsilon:
      return random.randint(0,self.num_actions-1)
    else :
      possible_values=[]
      for i in range(self.num_actions):
        possible_values.append(self.q_table[current_state][i])
      return np.argmax(possible_values)



  def update_policy(self,current_state,selected_action,reward,next_state,learning_rate):
    possible_values=[]
    # print(f'current_state={current_state}')
    # print(f'selected_action={selected_action}')
    # print(f'reward={reward}')
    # print(f'next_state={next_state}')
    # print(f'current_state={current_state}')
    for i in range(self.num_actions):
      possible_values.append(self.q_table[next_state][i])
    self.q_table[current_state][selected_action]=self.q_table[current_state][selected_action]+learning_rate*(reward-self.q_table[current_state][selected_action]+self.discount_factor*max(possible_values))
  def set_render(self):
    self.env.set_render()
  def generate_episode(self):
        history = []
        current_state = self.env.convert_location_to_state(self.env.reset()[0]["agent"])
        terminated = False
        count=0
        while not terminated and self.env.health > 15 and self.env.battery > 5:
            selected_action=np.argmax(self.q_table[current_state])
                
            # print(self.env._get_obs()['agent'])
            # print(selected_action)
            observation, cur_reward, terminated, truncated, info = self.env.step(
                selected_action
            )
            history.append([current_state, selected_action, cur_reward])
            next_state = self.env.convert_location_to_state(observation["agent"])
            # if self.env.is_target(observation["agent"]) or self.env.is_problem_maker(next_state):
                # print('episode ended with target')
                # print(next_state)

            if(next_state==current_state):
                count+=1
            current_state = next_state
            if terminated:
                history.append([current_state, selected_action, 1000])
        return history
  def q_learning(self):
    learning_rate=self.learning_rate
    reward_in_each_episode=[]
    episode_length=[]
    for _ in range(750):
      episode=[]

      current_state=self.env.convert_location_to_state(self.env.reset()[0]['agent'])
      reward=0
      terminated=False
      count=0
      horizon=1000
      episode.append(current_state)
      while not terminated and self.env.health>15 and self.env.battery>5:
        selected_action=self.select_action(current_state)
        observation, cur_reward, terminated, truncated, info=self.env.step(selected_action)
        self.update_policy(current_state,selected_action,reward,self.env.convert_location_to_state(observation['agent']),learning_rate)
        current_state=self.env.convert_location_to_state(observation['agent'])
        reward+=cur_reward
        count+=1
        episode.append(current_state)


      reward_in_each_episode.append(reward)
      episode_length.append(count)
      if self.with_decreasing_learning_rate:
        learning_rate=self.learning_rate/(_/50+1)
      self.epsilon=self.initial_epsilon/(1+_/250)
        
        

    return reward_in_each_episode,self.q_table,episode_length
