import numpy as np
import matplotlib.pyplot as plt
from GridWorld_env import *
from Q_Learning import *
from n_step_bt import *
import seaborn as sns

def mean_without_outliers(data, k=1.5):
    """
    Calculate the mean of the data while ignoring outliers using Tukey's fences.

    Parameters:
    - data (numpy.ndarray or list): Input data.
    - k (float): Tukey's fences constant. Typically set to 1.5.

    Returns:
    - float: Mean without outliers.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return np.mean(filtered_data)



SEED = 184


def get_optimal_policy(q_table):
    new_q_table = np.zeros((q_table.shape[0], q_table.shape[1]))
    for i in range(len(q_table)):
        index = np.argmax(q_table[i])
        new_q_table[i][index] = 1
    return new_q_table


def draw_curves(values):
    cmap = plt.get_cmap('tab10')
    random_colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    plt.figure()
    flag =True
    for _ in values:
        vals1=_[0]
        label1=_[1]
        mean_value = np.mean(vals1, axis=0)
        std_value = np.std(vals1, axis=0)
        upper_bound = mean_value + 1.96 * std_value / np.sqrt(len(vals1[0]))
        lower_bound = mean_value - 1.96 * std_value / np.sqrt(len(vals1[0]))
        plt.plot(mean_value, label=label1, color=random_colors[values.index(_)])
        plt.fill_between(
            np.arange(len(mean_value)),
            lower_bound,
            upper_bound,
            alpha=0.5,
            label=f"Confidence Interval for {label1}",
            color=random_colors[values.index(_)]
 )
        flag=False
    plt.legend()
    plt.show()

rewards=[]



gwe = GridWorldEnv(size=6)
np.random.seed(SEED)
random.seed(SEED)
start=0.1
best_agent=None
best_rewards=0
best_agent_1=None
best_rewards_1=0
best_agent_2=None
best_rewards_2=0
list_1=[[[],f'with decreasing learning rate from {1} to {1/(750/50+1)}'],[[],f'with decreasing learning rate from {start} to {start/(750/50+1)}'],[[],'without decreasing learning rate']]
for i in range(10):
    print(f"Agent{i+1}")
    gwe.reset()
    agent = Q_Learning(gwe, 1, 0.9, 0.1,with_decreasing_learning_rate=True)
    reward_in_each_episode, q_table,episodes_length = agent.q_learning()
    if(best_rewards<mean_without_outliers(reward_in_each_episode)):
        best_rewards=mean_without_outliers(reward_in_each_episode)
        best_agent=agent
    list_1[0][0].append(reward_in_each_episode)
    
for i in range(10):
    print(f"Agent{i+1}")
    gwe.reset()
    agent = Q_Learning(gwe, 0.1, 0.9, 0.1,with_decreasing_learning_rate=True)
    reward_in_each_episode, q_table,episodes_length = agent.q_learning()
    if(best_rewards_1<mean_without_outliers(reward_in_each_episode)):
        best_rewards_1=mean_without_outliers(reward_in_each_episode)
        best_agent_1=agent
    list_1[1][0].append(reward_in_each_episode)


for i in range(10):
    print(f"Agent{i+1}")
    gwe.reset()
    agent = Q_Learning(gwe, 0.1, 0.9, 0.1,with_decreasing_learning_rate=False)
    reward_in_each_episode, q_tabl,episodes_lengthe = agent.q_learning()
    if(best_rewards_2<mean_without_outliers(reward_in_each_episode)):
        best_rewards_2=mean_without_outliers(reward_in_each_episode)
        best_agent_2=agent
    list_1[2][0].append(reward_in_each_episode)    

draw_curves(list_1)
reward=[]
ep_len=[]
for _ in range(10):
    gwe.reset()
    gwe.set_render()
    
    ep=best_agent.generate_episode()
    ep_len=len(ep)
    rew=0
    for x in ep:
        s,a,r=x
        rew+=r
    reward.append(rew)
print(f'mean reward for agent 1={np.mean(reward)}')
print(f'mean episode length for agent 1={np.mean(ep_len)}')
reward=[]
ep_len=[]
for _ in range(10):
    gwe.reset()
    gwe.set_render()
    
    ep=best_agent_1.generate_episode()
    ep_len=len(ep)
    rew=0
    for x in ep:
        s,a,r=x
        rew+=r
    reward.append(rew)
print(f'mean reward for agent 2={np.mean(reward)}')
print(f'mean episode length for agent 2={np.mean(ep_len)}')
reward=[]
ep_len=[]
for _ in range(10):
    gwe.reset()
    gwe.set_render()
    
    ep=best_agent_2.generate_episode()
    ep_len=len(ep)
    rew=0
    for x in ep:
        s,a,r=x
        
        rew+=r
    reward.append(rew)
print(f'mean reward for agent 3={np.mean(reward)}')
print(f'mean episode length for agent 3={np.mean(ep_len)}')
# draw_curves(list_1)
        

