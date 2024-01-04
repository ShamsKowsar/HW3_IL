import numpy as np
import matplotlib.pyplot as plt
from GridWorld_env import *
from nSARSA import *
from Q_Learning import *
from n_step_bt import *
from SARSA import *
import seaborn as sns


SEED = 184
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




def get_optimal_policy(q_table):
    new_q_table = np.zeros((q_table.shape[0], q_table.shape[1]))
    for i in range(len(q_table)):
        index = np.argmax(q_table[i])
        new_q_table[i][index] = 1
    return new_q_table


def draw_curves(values):
    cmap = plt.get_cmap("tab10")
    random_colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    plt.figure()
    flag = True
    for _ in values:
        vals1 = _[0]
        label1 = _[1]
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
            color=random_colors[values.index(_)],
        )
        flag = False
    plt.legend()
    plt.show()


rewards = []

ba_s=[None for _ in range(4)]
br_s=[0 for _ in range(4)]
gwe = GridWorldEnv(size=6)
np.random.seed(SEED)
random.seed(SEED)
list1 = [[[],'SARSA']]
for i in range(10):
    print(f"Agent{i+1}")
    gwe.reset()

    agent = SARSA(gwe, 0.1, 0.9, 0.1)
    reward_in_each_episode, q_table, episode_len = agent.SARSA()

    list1[0][0].append(reward_in_each_episode)
    if mean_without_outliers(reward_in_each_episode)>br_s[0]:
        br_s[0]=mean_without_outliers(reward_in_each_episode)
        ba_s[0]=agent

for _ in range(3):
    n = 10**(_)

    list1.append([[], f"n step backup tree,n={n}"])

    for i in range(10):
        print(f"Agent{i+1}")
        gwe.reset()

        agent = NSTBT(gwe, 0.1, 0.9, 0.1, n)
        (
            reward_in_each_episode,
            q_table,
            policy,
            len_ep
            
        ) = agent.n_step_backup_tree()
        if mean_without_outliers(reward_in_each_episode)>br_s[0]:
            br_s[_+1]=mean_without_outliers(reward_in_each_episode)
            ba_s[_+1]=agent
        list1[_+1][0].append(reward_in_each_episode)
draw_curves(list1)
for z in range(4):
    reward=[]
    ep_len=[]
    for _ in range(10):
        gwe.reset()
        gwe.set_render()
        
        ep=ba_s[z].generate_episode()
        ep_len=len(ep)
        rew=0
        for x in ep:
            s,a,r=x
            
            rew+=r
        reward.append(rew)
print(f'mean reward for agent {z}={np.mean(reward)}')
print(f'mean episode length for agent {z}={np.mean(ep_len)}')

