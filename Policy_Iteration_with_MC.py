
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




import gym
from gym import spaces
import pygame
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from On_MC import *
class Policy_IterationWMC:
    def __init__(
        self,
        env,
        discount_factor,
        epsilon,
        with_decreasing_learning_epsilon=False,
    ):
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_states = env.get_state_size()
        self.num_actions = env.get_action_size()
        self.policy = np.random.rand(env.get_state_size(), env.get_action_size())
    
        agent = On_MC(self.env, self.discount_factor, self.epsilon,with_decreasing_learning_epsilon=False)
        reward_in_each_episode, q_table,count = agent.On_MC()
        self.values=np.random.rand(self.num_states)
        self.On_MC()
        # self.generate_policy()
        self.policy_stable = False
        self.value_stable = False
    def generate_policy(self):
        for i in range(self.env.size):
            index=2
            while index==2:
                index=random.randint(0,self.num_actions-1)
            self.policy[i][index]=1
        for i in range(self.env.size):
            index=0
            while index==0:
                index=random.randint(0,self.num_actions-1)
            self.policy[self.env.size*(self.env.size-1)+i][index]=1
        for i in range(self.env.size):
            index=1
            while index==1:
                index=random.randint(0,self.num_actions-1)
            self.policy[i*self.env.size+self.env.size-1][index]=1 
        for i in range(self.env.size):
            index=3
            while index==3:
                index=random.randint(0,self.num_actions-1)
            self.policy[i*self.env.size][index]=1        
      
    def given_policy(self, current_state):
        return np.argmax(self.policy[current_state])
    def calc_2(self,state,action):
        # print(state)
        direction = (self.env._action_to_direction[action])
        new_direction=direction+[0,0]
        value=0
        possible_next_states=[]
        # print('--------------------------------------------------------')
        # goes in chosen direction:
        new_location=np.clip([int(state/6),state%6] + new_direction, 0, self.env.size - 1)
        # print(new_location)
        if self.env.is_target(new_location):
          possible_next_states.append([self.env.convert_location_to_state(new_location),0.8,25])
          # print([self.env.convert_location_to_state(new_location),0.8,25])
        elif self.env.is_obstacle(new_location):
          possible_next_states.append([state,0.8,-1])
          # print([state,0.8,-1])
          
        else:
          possible_next_states.append([self.env.convert_location_to_state(new_location),0.8,-0.5])
          # print([self.env.convert_location_to_state(new_location),0.8,-0.5])
        # goes in reverse direction:
        new_location=np.clip([int(state/6),state%6]  - new_direction, 0, self.env.size - 1)
        # print(new_location)
        
        if self.env.is_target(new_location):
          possible_next_states.append([self.env.convert_location_to_state(new_location),0.1,25])
          # print([self.env.convert_location_to_state(new_location),0.1,25])
        elif self.env.is_obstacle(new_location):
          possible_next_states.append([state,0.1,-1])
          # print([state,0.1,-1])
          
        else:
          possible_next_states.append([self.env.convert_location_to_state(new_location),0.1,-0.5])
          # print([self.env.convert_location_to_state(new_location),0.1,-0.5])
        # doesn't move:
        new_location=np.clip([int(state/6),state%6] , 0, self.env.size - 1)
        # print(new_location)
        
        if self.env.is_target(new_location):
          # print('G')
          possible_next_states.append([self.env.convert_location_to_state(new_location),0.1,25])
          # print([self.env.convert_location_to_state(new_location),0.1,25])
        elif self.env.is_obstacle(new_location):
          # print('O')
          # print(state)
          # print()
          possible_next_states.append([state,0.1,-1])
          # print([state,0.1,-1])
          
        else:
          # print('N')
          possible_next_states.append([self.env.convert_location_to_state(new_location),0.1,-0.5])
          # print([self.env.convert_location_to_state(new_location),0.1,-0.5])
        # print('--------------------------------------------------------')
        
        value=0
        for _ in possible_next_states:
          state,prob,reward=_
          value+=prob*(reward+self.discount_factor*self.values[int(state)])
        return value
    
    def update_policy(self, state):
        possible_values = []
        for action in range(self.num_actions):
            possible_values.append(self.calc_2(state, action))

        index = np.argmax(possible_values)
        for i in range(self.num_actions):
            if i == index:
                self.policy[state][i] = 1
            else:
                self.policy[state][i] = 0

    def generate_episode(self):
        history = []
        current_state = self.env.convert_location_to_state(self.env.reset()[0]["agent"])
        terminated = False
        count=0
        while not terminated and self.env.health > 15 and self.env.battery > 5:
            if count<20:
                selected_action = self.given_policy(current_state)
            else:
                selected_action=self.soft_policy(current_state)
                count=0
                
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
    def soft_policy(self,current_state):
        _=random.uniform(0,1)
        if _<self.epsilon:
            return random.randint(0,self.num_actions-1)
        else :
            possible_values=[]
            possible_max_values=[]
            for i in range(self.num_actions):
                possible_values.append(self.values[current_state])
            for value in possible_values:
                if value==max(possible_values):
                    possible_max_values.append(value)

            if len(possible_max_values)>1:
                index=random.randint(0,len(possible_max_values)-1)
                u=0
                for i in range(self.num_actions):
                    if self.values[current_state]==max(possible_values):
                        if(u==index):
                            return i
                        u+=1


                return random.randint(0,self.num_actions-1)
            return np.argmax(possible_values)
  
    def check_first_visit(self, episode, i, sar):
        s, a, r = sar
        for _ in range(i):
            if episode[_][0] == s and episode[_][1] == a:
                return True
        return False

    def policy_eval(self):
        delta=np.inf
        while delta>0.000000001:
            delta=0
            for state in range(self.num_states):
                temp_v=self.values[state]
                self.values[state]=self.calc_2(state,np.argmax(self.policy[state]))  
                delta=max(delta,abs(temp_v-self.values[state]))                  
            
    def On_MC(self):
        returns = [[] for _ in range(self.num_states)]
        els=[]
        delta=0
        for _ in range(750):
            episode = self.generate_episode()
            G = 0
            T = len(episode)
            reward = 0
            for i in range(T):

                t=T-i-1
                s, a, r = episode[t]
                G = self.discount_factor * G + r

                if not self.check_first_visit(episode, t, episode[T - i - 1]):
                    returns[s].append(G)
                    old_value=self.values[s]
                    new_value=np.mean(returns[s])
                    if(self.env.is_obstacle([int(s/6),s%6])):
                        self.values[s] = -100
                        
                        
                    self.values[s] = new_value
                    if self.env.is_target([int(s/6),s%6]):
                        self.values[s]=1000
                    delta=max(delta,abs(old_value-new_value))
                    if(delta<0.05):
                        self.value_stable=True

        return self.values
            
    def policy_improvement(self):
        # for _ in range(self.num_states):
        #     print(f"state={_} and value= {self.values[_]}")
        self.policy_stable = True
        for state in range(self.num_states):
            old_action = np.argmax(self.policy[state])
            self.update_policy(state)
            new_action = np.argmax(self.policy[state])
            self.update_policy(state)
            if old_action != new_action:
                self.policy_stable = False

    def fit(self):
        count=0
        rs=[]
        len_ep=[]
        # self.MC_evaluation()
    
        while (not self.policy_stable ) or count<50:
            ep=self.generate_episode()
            vall=0
            for x in ep:
                s,a,r=x
                if(r!=1000):
                    vall+=r
            rs.append(rs)
            self.policy_eval()
            self.policy_improvement()
            count+=1
            print(count)
        return rs, self.values,count
    def fit_2(self):
        count=0
        rs=[]
        len_ep=[]
    
        while (not self.policy_stable ) and count<50:
            ep=self.generate_episode()
            vall=0
            for x in ep:
                s,a,r=x
                if(r!=1000):
                    vall+=r
            rs.append(rs)
            if (not self.policy_stable ):
                self.On_MC()
                self.policy_improvement()
            count+=1
            print(count)
        return rs, self.values,count

