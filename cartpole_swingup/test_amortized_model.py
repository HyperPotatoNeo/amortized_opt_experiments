import numpy as np
import torch
import gym
import modified_gym_cartpole_swingup
import fully_amortized_model
import gym_cartpole_swingup
import os

env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
env.reset()

T = 40
actions = []
done = False
eps = 10
obs_dim = 5
TEST_LSTM = True

if TEST_LSTM:
	model = torch.load('fully_amortized_models/LSTM_Dagger_5')
	policy = fully_amortized_model.LSTM_direct_policy(env=env, action_dim=1, T=1, model=model, N=1)
else:
	model = torch.load('fully_amortized_models/MLP_Dagger_1')
	policy = fully_amortized_model.MLP_direct_policy(obs_dim=obs_dim, action_dim=1, T=T, model=model)

with torch.no_grad():
	for i in range(eps):
		obs = env.reset()
		for j in range(500):
			if TEST_LSTM:
				actions = policy(env.state, batch_size=1)
			else:
				actions = policy(obs)
				action = actions[0, 0]
				obs, rewards, done, info = env.step(action)
				if done:
					break
			env.render()