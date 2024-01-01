
import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

class Tree_Backup():
  def __init__(self,env,n,learning_rate,discount_factor,epsilon,with_decreasing_learning_rate=False) :
    self.env=env
    self.learning_rate=learning_rate
    self.discount_factor=discount_factor
    self.epsilon=epsilon
    self.num_states=env.get_state_size()
    self.num_actions=env.get_action_size()
    self.n=n
    self.q_table=np.zeros((env.get_state_size(),env.get_action_size()))
  def select_action(self,current_state):
    _=random.uniform(0,1)
    if _<self.epsilon:
      return random.randint(0,self.num_actions-1)
    else :
      possible_values=[]
      possible_max_values=[]
      for i in range(self.num_actions):
        possible_values.append(self.q_table[current_state][i])
      for value in possible_values:
        if value==max(possible_values):
          possible_max_values.append(value)

      if len(possible_max_values)>1:
        index=random.randint(0,len(possible_max_values)-1)
        u=0
        for i in range(self.num_actions):
          if self.q_table[current_state][i]==max(possible_values):
            if(u==index):
              return i
            u+=1


        return random.randint(0,self.num_actions-1)
      return np.argmax(possible_values)


  def update_policy(self,current_state,selected_action,reward,next_state,next_state_selected_action,learning_rate):
    possible_values=[]
    for i in range(self.num_actions):
      possible_values.append(self.q_table[next_state][i])
    # print(self.q_table[current_state][selected_action])

    self.q_table[current_state][selected_action]=self.q_table[current_state][selected_action]+learning_rate*(reward-self.q_table[current_state][selected_action]+self.discount_factor*self.q_table[next_state][next_state_selected_action])
    # print(self.q_table[current_state][selected_action])

  def Tree_Backup(self):
    learning_rate=self.learning_rate
    reward_in_each_episode=[]

    for _ in range(750):
      episode=[]
      history=[]
      TNSB_episode=""
      current_state=self.env.convert_location_to_state(self.env.reset()[0]['agent'])
      selected_action=self.select_action(current_state)
      t=0
      reward=0
      terminated=False
      episode.append(current_state)
      observation, cur_reward, terminated, truncated, info=self.env.step(selected_action)
      history.append([current_state,selected_action,cur_reward])
      while not terminated and self.env.health>15 and self.env.battery>5:

        for i in range(self.n):
          selected_action=self.greedy_wrt_q(current_state)
          observation, cur_reward, terminated, truncated, info=self.env.step(selected_action)
          history.append([current_state,selected_action,cur_reward])
          next_state=self.env.convert_location_to_state(observation['agent'])
          current_state=next_state
        for j in range(t,t+self.n):
          s,a,r=history[j]
          G=calculate_g(history)
          self.q_table[s][a]=self.q_table[s][a]+self.learning_rate(G-self.q_table[s][a])
        selected_action=self.select_action(s)
        t=0
        reward=0
        terminated=False
        episode.append(current_state)
        observation, cur_reward, terminated, truncated, info=self.env.step(selected_action)
        history.append([current_state,selected_action,cur_reward])

        observation, cur_reward, terminated, truncated, info=self.env.step(selected_action)
        next_state=self.env.convert_location_to_state(observation['agent'])
        next_state_selected_action=self.select_action(next_state)
        SARSA_episode+=f"{current_state},{selected_action},{reward},{next_state},{next_state_selected_action}"
        self.update_policy(current_state,selected_action,reward,next_state,next_state_selected_action,learning_rate)
        current_state=next_state
        selected_action=next_state_selected_action
        reward+=cur_reward
        count+=1
        episode.append(current_state)
        # print(f"{current_state},{cur_reward},{selected_action}")

      reward_in_each_episode.append(reward)
      # learning_rate=self.learning_rate/(_/100+1)

      # print(episode)
      # print(f'end of episode{_+1}')
    # print(self.q_table)
    return reward_in_each_episode,self.q_table
