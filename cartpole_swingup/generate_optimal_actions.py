# coding: utf-8
import gym
import numpy as np
import torch
import gym_cartpole_swingup
import modified_gym_cartpole_swingup

def reset_environment(env, state):
    env.reset()
    env.state = state

env = gym.make("TorchCartPoleSwingUp-v0")
data = np.load('dagger_data/data_1.npy')
mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")

T = 70
T_model = 40
actions = []
state_dim = 4
obs_dim = 5

######TO DO#########
###FIND WAY TO EFFICIENTLY GENERATE OPTIMAL ACTIONS USING GRADIENT DESCENT WITHOUT WARMSTART###
###JUST USE PPO AGENT IF NOT POSSIBLE####

print('GENERATING OPTIMAL ACTIONS:')

for i in range(T):
    actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))

data_actions = np.zeros((data.shape[0],state_dim+obs_dim+T))

for k in range(data.shape[0]):
    data_actions[k,:state_dim+obs_dim] = data[k]

    for j in range(T):
        actions[j] = torch.tensor(env.action_space.sample(), requires_grad=True)
    optimizer = torch.optim.Adam(actions, lr=0.01)

    for i in range(20):
        optimizer.zero_grad()
        reset_environment(mpc_env, torch.as_tensor(data[k,:state_dim]))
        rewards = 0
        for t in range(T):
            mpc_env.state, reward, done, _ = mpc_env.mpc_step(mpc_env.state, actions[t])
            rewards -= reward
        rewards.backward(retain_graph=True)
        optimizer.step()
    for j in range(T_model):
        data_actions[k,state_dim+obs_dim+j] = actions[j].detach().numpy()[0]
    
np.save('dagger_data/data_actions.npy',data_actions)