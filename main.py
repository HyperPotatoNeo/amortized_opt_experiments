# coding: utf-8
import gym
import numpy as np
import torch
import gym_cartpole_swingup

# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("TorchCartPoleSwingUp-v0")
env.reset()
env.state = torch.tensor([-0.5, 0, np.pi, 0], requires_grad=False)
mpc_env = gym.make("TorchCartPoleSwingUp-v0")

T = 70
actions = []
done = False

def reset_environment(env, state):
    env.reset()
    env.state = state

for i in range(T):
    actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))

while not done:
    actions = actions[1:]
    actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
    optimizer = torch.optim.Adam(actions, lr=0.01)

    for i in range(10):
        optimizer.zero_grad()
        reset_environment(mpc_env, env.state)
        rewards = 0
        for t in range(T):
            mpc_env.state, reward, done, _ = mpc_env.mpc_step(mpc_env.state, actions[t])
            rewards -= reward
        rewards.backward(retain_graph=True)
        optimizer.step()

    env.state, rew, done, info = env.mpc_step(env.state, actions[0])
    env.render()
