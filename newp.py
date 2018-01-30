#Performed a simple weighted cross entropy method.
#Simply copy pasted the following code it should work!

import gym
from gym import wrappers
import matplotlib.pyplot as plt
import csv



def play(num_episodes, num_steps, policy, update=None):
    time_steps = []
    for i_episode in range(num_episodes):
        observation = env.reset()
        states, actions, rewards = [], [], []
        r = 0
        states.append(observation)
        for t in range(num_steps):
            action = policy(observation)
            
            observation, reward, done, info = env.step(action)
            r += reward
            
            states.append(observation)
            actions.append(action)
            rewards.append(reward)
                
            if done:
                break
        
        if update:
            update(actions, states, rewards)
        
        states.append(observation)
        
        time_steps.append(t)

    #env.close()
    #w, b = best_params
    return time_steps, r
  
import operator
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import wrappers

#Cross-Entropy
np.random.seed(0)

ds = [3, 100, 50, 1]
dim = 0
for i in range(len(ds)-1):
    dim += (ds[i]+1)*(ds[i+1])

print(dim)

curr_mean = 0
curr_std = np.ones(dim)
it = 0
        
def policy_cem(obs):
    out = obs.reshape(1, 3)
    W = Ws[it]
    
    ind = 0
    for i in range(len(ds)-1):
        out = out.reshape(1, -1)
        
        d1 = ds[i]
        d2 = ds[i+1]
        w = W[ind:ind+d1*d2].reshape(d1, d2)
        ind = ind+d1*d2
        b = W[ind:ind+d2]
        ind = ind + d2
        out = out.dot(w) + b
        if ind != len(W):
            out = np.tanh(out)
    
    p = out[0]
    if p >= -2 and p <= 2:
        return p
    if p < -2:
        return [-2]
    if p > 2:
        return [2]
    
env = gym.make('Pendulum-v0')
env = wrappers.Monitor(env, '/tmp/pendulum-experiment-1',force=True)
scores = []
std = []
mean = []
alpha = 0.5

for i in range(1000):
    
    Ws = (np.random.randn(100, dim))*curr_std + curr_mean
    b = {}
    for j in range(100):
        
        it = j
        sc, r = play(1, 500, policy_cem)
        b[j] = -r
    
    scores.append(np.mean([-v for i, v in b.items()]))
    print(scores[-1])
    sorted_b = sorted(b.items(), key=operator.itemgetter(1))
    idx = [k for k, v in sorted_b[:20]]
    v = np.array([v for k, v in sorted_b[:20]]).reshape(-1, 1)
    
    curr_mean = curr_mean*(1-alpha) + alpha*np.sum(Ws[idx]*v, axis=0)/np.sum(v)
    curr_std = curr_std*(1-alpha) + alpha*np.sqrt(np.sum(v*(Ws[idx]-curr_mean)**2, axis=0)/np.sum(v)) #+ np.max(10-i/100, 0)
    
    if np.sqrt(np.sum(curr_std**2)) < 1e-3:
        break
    
    std.append(curr_std[0])
    mean.append(curr_mean[0])

env.close()

example_file = open('newp.csv','w', newline='')
writer=csv.writer(example_file, delimiter =',')
writer.writerow(['alpha','episode','score'])
writer.writerow([alpha, num_episodes, scores ])

writer.close()

#plt.plot(std)
#plt.plot(mean)
plt.plot(scores)
plt.show()