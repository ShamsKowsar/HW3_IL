
import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

from GridWorld_env import *
from SARSA import *
from Q_Learning import *
from On_MC import *
from TNSB import *






SEED=184


def get_optimal_policy(q_table):
  new_q_table=np.zeros((q_table.shape[0],q_table.shape[1]))
  for i in range(len(q_table)):
    index=np.argmax(q_table[i])
    new_q_table[i][index]=1
  return new_q_table


gwe=GridWorldEnv(size=6)
for i in range(10):
  print(f'Agent{i+1}')
  gwe.reset()
  qq=SARSA(gwe,0.1,0.9,0.1,False)
  res,q_table=(qq.SARSA())
  # if qq.has_learning_converged():
  #   print('learning occured')
  print(q_table)
  optimal_policy=(get_optimal_policy(q_table))
  index=0
  for row in optimal_policy:
    print(f'row={index}-action={"down" if row[0]==1 else "right" if row[1]==1 else"up" if row[2]==1 else "left"}')
    index+=1

  print(res)
  plt.plot(res)
  plt.show()

















gwe=GridWorldEnv(size=6)
for i in range(10):
  print(f'Agent{i+1}')
  gwe.reset()
  qq=On_MC(gwe,0.9,0.1)
  res,q_table=(qq.On_MC())
  # if qq.has_learning_converged():
  #   print('learning occured')
  print(q_table)
  optimal_policy=(get_optimal_policy(q_table))
  index=0
  for row in optimal_policy:
    print(f'row={index}-action={"down" if row[0]==1 else "right" if row[1]==1 else"up" if row[2]==1 else "left"}')
    index+=1

  print(res)
  plt.plot(res)
  plt.show()

    
        