
# import gym
# from gym import spaces
# import pygame
# import numpy as np
# from collections import deque
# import random
# import matplotlib.pyplot as plt

# class SARSA():
#   def __init__(self,env,learning_rate,discount_factor,epsilon,with_decreasing_learning_rate=False) :
#     self.env=env
#     self.learning_rate=learning_rate
#     self.discount_factor=discount_factor
#     self.epsilon=epsilon
#     self.num_states=env.get_state_size()
#     self.num_actions=env.get_action_size()
#     self.q_table=np.zeros((env.get_state_size(),env.get_action_size()))
#   def select_action(self,current_state):
#     _=random.uniform(0,1)
#     if _<self.epsilon:
#       return random.randint(0,self.num_actions-1)
#     else :
#       possible_values=[]
#       possible_max_values=[]
#       for i in range(self.num_actions):
#         possible_values.append(self.q_table[current_state][i])
#       for value in possible_values:
#         if value==max(possible_values):
#           possible_max_values.append(value)

#       if len(possible_max_values)>1:
#         index=random.randint(0,len(possible_max_values)-1)
#         u=0
#         for i in range(self.num_actions):
#           if self.q_table[current_state][i]==max(possible_values):
#             if(u==index):
#               return i
#             u+=1


#         return random.randint(0,self.num_actions-1)
#       return np.argmax(possible_values)


#   def update_policy(self,current_state,selected_action,reward,next_state,next_state_selected_action,learning_rate):
#     # possible_values=[]

#     self.q_table[current_state][selected_action]=self.q_table[current_state][selected_action]+learning_rate*(reward-self.q_table[current_state][selected_action]+self.discount_factor*self.q_table[next_state][next_state_selected_action])
#     # print(self.q_table[current_state][selected_action])

#   def SARSA(self):
#     learning_rate=self.learning_rate
#     reward_in_each_episode=[]

#     for _ in range(750):
#       episode=[]
#       SARSA_episode=""

#       current_state=self.env.convert_location_to_state(self.env.reset()[0]['agent'])
#       selected_action=self.select_action(current_state)

#       reward=0
#       terminated=False
#       count=0
#       episode.append(current_state)
#       while not terminated and self.env.health>15 and self.env.battery>5:

#         observation, cur_reward, terminated, truncated, info=self.env.step(selected_action)
#         next_state=self.env.convert_location_to_state(observation['agent'])
#         next_state_selected_action=self.select_action(next_state)
#         SARSA_episode+=f"{current_state},{selected_action},{reward},{next_state},{next_state_selected_action}"
#         self.update_policy(current_state,selected_action,reward,next_state,next_state_selected_action,learning_rate)
#         current_state=next_state
#         selected_action=next_state_selected_action
#         reward+=cur_reward
#         count+=1
#         episode.append(current_state)
#         # print(f"{current_state},{cur_reward},{selected_action}")

#       reward_in_each_episode.append(reward)
#       # learning_rate=self.learning_rate/(_/100+1)

#       # print(episode)
#       # print(f'end of episode{_+1}')
#     # print(self.q_table)
#     return reward_in_each_episode,self.q_table
import gym
import numpy as np
import random

class NSarsa:
    def __init__(self, env, learning_rate, discount_factor, epsilon, n):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_states = env.get_state_size()
        self.num_actions = env.get_action_size()
        self.q_table = np.zeros((env.get_state_size(), env.get_action_size()))
        self.n = n
        self.policy=np.random.rand(self.num_states,self.num_actions)

    def select_action(self, current_state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[current_state])

    def update_policy(self,state):
        epsilon_greedy_action=np.argmax(self.q_table[state])
        for i in range(self.num_actions):
            if i== epsilon_greedy_action:
                self.policy[state][i]=1-self.epsilon+self.epsilon/self.num_actions
            else:
                self.policy[state][i]=self.epsilon/self.num_actions
           

        
    def calc_g(self,states,actions,rewards,tau,T):
        index=min(tau+self.n,T)
        needed_rewards=rewards[tau+1:index+1]
        if len(needed_rewards)<self.n:
            z=(self.n-len(needed_rewards))
            new_arr=rewards[:z]
            needed_rewards=np.concatenate((needed_rewards,new_arr))
        # print(len(needed_rewards))
        # print(range(tau + 1, index + 1))
        G = np.sum(self.discount_factor ** i * needed_rewards[i - (tau + 1)] for i in range(tau + 1, index + 1))
        if tau+self.n<T:
            end=(tau+self.n)%(self.n+1)
            G+=self.discount_factor**self.n*self.q_table[states[end]][actions[end]]
        return G
        
        
    def n_step_sarsa(self):
        reward_in_each_episode = []
        episode_length=[]

        for _ in range(750):
            states=[0 for i in range(self.n+1)]
            actions=[0 for i in range(self.n+1)]
            rewards=[0 for i in range(self.n+1)]
            current_state = self.env.convert_location_to_state(self.env.reset()[0]['agent'])
            selected_action = self.select_action(current_state)
            states[0]=(current_state)
            actions[0]=(selected_action)
            T=np.inf
            t=0
            reward=0
            while True:
                current=t%(self.n+1)
                next=(t+1)%(self.n+1)
                if(t<T):
                    observation, cur_reward, terminated, _, info = self.env.step(actions[current])
                    states[next]=(self.env.convert_location_to_state(observation['agent']))
                    reward+=cur_reward
                    rewards[next]=(cur_reward)
                    if terminated:
                        T=t+1
                    else:
                        actions[next]= self.select_action(states[next])
                tau=t-self.n+1
                if tau>=0:
                    G=self.calc_g(states,actions,rewards,tau,T)
                    start=tau%(self.n+1)
                    self.q_table[states[start]][actions[start]]=self.q_table[states[start]][actions[start]]+self.learning_rate*(G-self.q_table[states[start]][actions[start]])
                    self.update_policy(states[start])
                if tau==(T-1):
                    # for i in range(self.num_actions):
                    #     self.q_table[states[-1]][i]=1000
                    # # print(self.q_table)
                    break
                
                t+=1
            
            reward_in_each_episode.append(reward)
            episode_length.append(T)
            
            self.epsilon=1/(_+1)/10
        return reward_in_each_episode, self.q_table,self.policy,episode_length
