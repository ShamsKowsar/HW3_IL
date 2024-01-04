
import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

class On_MC():
  
  def __init__(self,env,discount_factor,epsilon,with_decreasing_learning_epsilon=False,reduction_factor=1) :
    self.env=env
    self.discount_factor=discount_factor
    self.epsilon=epsilon
    self.num_states=env.get_state_size()
    self.num_actions=env.get_action_size()
    self.q_table=np.zeros((env.get_state_size(),env.get_action_size()))
    self.policy=np.zeros((env.get_state_size(),env.get_action_size()))
    self.with_decreasing_learning_epsilon=with_decreasing_learning_epsilon
    self.reduction_factor=reduction_factor
  def soft_policy(self,current_state):
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
  
  def generate_episode(self):
    history=[]
    current_state=self.env.convert_location_to_state(self.env.reset()[0]['agent'])
    terminated=False
    while not terminated and self.env.health>15 and self.env.battery>5:
      selected_action=self.soft_policy(current_state)
      observation, cur_reward, terminated, truncated, info=self.env.step(selected_action)
      history.append([current_state,selected_action,cur_reward])
      next_state=self.env.convert_location_to_state(observation['agent'])
      next_state_selected_action=self.select_action(next_state)
      current_state=next_state
    return history

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


  def update_policy(self,s,choosen_best_action):
    for i in range(self.num_actions):
      if i!=choosen_best_action:
        self.policy[s][i]=1-self.epsilon+self.epsilon/self.num_actions
    self.policy[s][choosen_best_action]=self.epsilon/self.num_actions

  def choose_best_action(self,state):
    possible_values=[]
    for i in range(self.num_actions):
      possible_values.append(self.q_table[state][i])
    return np.argmax(possible_values)
  def check_first_visit(self,episode,i,sar):
    s,a,r=sar
    for _ in range(i):
      if episode[_][0]==s and episode[_][1]==a:
        return False
    return True
  def On_MC(self):
    reward_in_each_episode=[]
    episoed_length=[]
    returns= ([[[] for _ in range(self.num_actions)] for _ in range(self.num_states)])
    
    for _ in range(750):
      episode=self.generate_episode()
      episoed_length.append(len(episoed_length))
      G=0
      T=len(episode)
      reward=0
      for i in range(T):
        s,a,r=episode[T-i-1]
        G=self.discount_factor*G+episode[T-i-1][-1]
        if self.check_first_visit(episode,i,episode[T-i-1]):
          # print(s)
          # print(a)
          returns[s][a].append(G)
          self.q_table[s][a]=np.mean(returns[s][a])
          best_action=self.choose_best_action(s)
          self.update_policy(s,best_action)
        reward+=r
      reward_in_each_episode.append(reward)
      if self.with_decreasing_learning_epsilon:
        self.epsilon=1/(1+_/self.reduction_factor)




    return reward_in_each_episode,self.q_table,episoed_length
