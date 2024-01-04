
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

class NSTBT:
    def __init__(self, env, learning_rate, discount_factor, epsilon, n):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_states = env.get_state_size()
        self.num_actions = env.get_action_size()
        self.q_table = np.random.rand(env.get_state_size(), env.get_action_size())
        self.n = n
        self.policy=np.random.rand(self.num_states,self.num_actions)
        for state in range(self.num_states):
            valval=sum(self.policy[state])
            for action in range(self.num_actions):
                self.policy[state][action]/=valval
        for state in range(self.num_states):
             if(abs(np.sum(self.policy[state])-1)>0.00001):
                print('BITCH')
    def select_action(self, current_state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.policy[current_state])

    def update_policy(self,state):
 
        epsilon_greedy_action=np.argmax(self.q_table[state])
        for i in range(self.num_actions):
            if i== epsilon_greedy_action:
                self.policy[state][i]=1-self.epsilon+self.epsilon/self.num_actions
            else:
                self.policy[state][i]=self.epsilon/self.num_actions

        if(abs(np.sum(self.policy[state])-1)>0.00001):
            print('BITCH')
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
    def calc_val(self,state):
        # state=states[index]
        value=0
        for i in range(self.num_actions):
            value+=self.policy[state][i]*self.q_table[state][i]
                    
        return value
    def calc_val_ign_action(self,action,state):
        value=0
        for i in range(self.num_actions):
            if i!=action:
                value+=self.policy[state][i]*self.q_table[state][i]
                    
        return value
  
        
    def n_step_backup_tree(self):
        reward_in_each_episode = []
        len_ep=[]

        for _ in range(750):
            
            states=[0 for i in range(self.n+1)]
            actions=[0 for i in range(self.n+1)]
            rewards=[0 for i in range(self.n+1)]
            current_state = self.env.convert_location_to_state(self.env.reset()[0]['agent'])
            # selected_action = np.random.randint(0,self.num_actions-1)
            # selected_action = np.random.randint(0,self.num_actions-1)
            states[0]=(current_state)
            selected_action = np.argmax(self.q_table[states[-1]])
            actions[0]=(selected_action)
            T=np.inf
            t=0
            reward=0
            while True:
                current=t%(self.n+1)
                next=(t+1)%(self.n+1)
                if(t<T):
                    observation, cur_reward, terminated, kkk, info = self.env.step(actions[current])
                    states[next]=(self.env.convert_location_to_state(observation['agent']))
                    reward+=cur_reward
                    rewards[next]=(cur_reward)
                    actions[next]= np.argmax(self.q_table[states[-1]])

                    
                    if terminated:
                        T=t+1
                    
                tau=t-self.n+1
                if tau>=0:
                    max_t=(min(t+1,T))
                    conv_max_t=max_t %(self.n+1)
                    G = rewards[conv_max_t] + (max_t == T) * self.discount_factor * (self.calc_val(states[conv_max_t]))

                    for k in range(max_t-1,tau,-1):
                        index=k%(self.n+1)
                        s,a,r=states[index],actions[index],rewards[index]
                        G = r + self.discount_factor * (self.calc_val_ign_action(a, s) + self.policy[s][a] * G)
                    s,a=states[tau%(self.n+1)],actions[tau%(self.n+1)]
                    self.q_table[s][a]+=self.learning_rate*(G-self.q_table[s][a])
                    self.update_policy(s)
                if tau==(T-1):
                    # for aa in range(self.num_actions):
                    #     if self.env.is_target([int(s/6),s%6]):
                    #         self.q_table[states[-1]][aa]=1000
                            
                    
                    break
                
                t+=1
            
            reward_in_each_episode.append(reward)
            len_ep.append(T)
            self.epsilon=1/(_/50+1)
            
            # self.epsilon=1/(_+1)/50

        return reward_in_each_episode, self.q_table,self.policy,len_ep
