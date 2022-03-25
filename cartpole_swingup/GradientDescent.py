# coding: utf-8
import gym
import numpy as np
import torch
import gym_cartpole_swingup
import modified_gym_cartpole_swingup

# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("CartPoleSwingUp-v0")
env.reset()
mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")

T = 70
actions = []
done = False

for i in range(T):
    actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))

while not done:
    actions = actions[1:]
    actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
    optimizer = torch.optim.Adam(actions, lr=0.01)

    for i in range(10):
        optimizer.zero_grad()
        mpc_env.mpc_reset(1, torch.as_tensor(env.state))
        rewards = 0
        for t in range(T):
            obs, reward, done, _ = mpc_env.step(actions[t])
            rewards -= reward
        rewards.backward()
        optimizer.step()
    obs, rew, done, info = env.step(actions[0].detach().numpy())
    env.render()
