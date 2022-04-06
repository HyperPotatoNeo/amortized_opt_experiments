import gym
import torch
import gym_cartpole_swingup
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import numpy as np
import os.path

class NoisyPolicy:
    def __init__(self, policy_model, action_noise_std=0.3):
        self.policy_model = policy_model
        self.action_noise_std = action_noise_std

    def __call__(self, obs):
        res, _states = self.policy_model.predict(obs)
        res += np.random.normal(loc=0.0, scale=self.action_noise_std, size=1)[0]
        return res


def learn_PPO_model(env, timesteps, name):
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(timesteps)
    model.save('policy_models/'+name)

    for r in range(10):
        obs = env.reset()
        for _ in range(400):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()


def collect_feasible_data(env, policy, eps=100, ep_len=300, state_dim=4, filename='feasible_dynamics.csv'):
    data = np.zeros((0, 2*state_dim+1))
    for i in range(eps):
        obs = env.reset()
        prev_state = env.state
        done = False
        t = 0
        while (not done and t<ep_len):
            t += 1
            action = policy(obs)
            obs, rewards, done, info = env.step(action)
            cur_state = env.state
            states = np.hstack([prev_state,cur_state,1.0])
            data = np.vstack((data, states))
            prev_state = cur_state
            if(i%20==0):
                env.render()

    if(os.path.exists('feasibility_data/'+filename)):
        data_prev = pd.read_csv('feasibility_data/'+filename, header=None).values
        data = np.vstack([data,data_prev])

    pd.DataFrame(data).to_csv('feasibility_data/'+filename, header=False, index=False)


def collect_infeasible_data(env, policy, noise=0.6, eps=300, ep_len=450, state_dim=4, filename='infeasible_dynamics.csv'):
    data = np.zeros((0, 2*state_dim+1))
    for i in range(eps):
        obs = env.reset()
        prev_state = env.state
        done = False
        t = 0
        while (not done and t<ep_len):
            t += 1
            action = policy(obs)
            obs, rewards, done, info = env.step(action)
            cur_state = env.state
            prev_state_r = np.random.normal(prev_state, noise)
            cur_state_r = np.random.normal(cur_state, noise)
            states = np.hstack([prev_state_r,cur_state_r,0.0])
            data = np.vstack((data, states))
            prev_state = cur_state
            if(i%20==0):
                env.render()
    if(os.path.exists('feasibility_data/'+filename)):
        data_prev = pd.read_csv('feasibility_data/'+filename, header=None).values
        data = np.vstack([data,data_prev])

    pd.DataFrame(data).to_csv('feasibility_data/'+filename, header=False, index=False)


if __name__=='__main__':
    env = gym.make("CartPoleSwingUp-v0")
    learn_policy = True
    collect_feasible_dynamics = True
    collect_infeasible_dynamics = False

    if(learn_policy):
        policy = learn_PPO_model(env, 200000, 'ppo_mlp')
    else:
        policy = PPO.load('policy_models/ppo_mlp')

    if(collect_feasible_dynamics):
        print('Collecting feasible data:')  
        for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
            print('Noise: ', noise)
            noisy_policy = NoisyPolicy(policy, noise)
            collect_feasible_data(env=env, policy=noisy_policy, eps=50, ep_len=450, state_dim=4, filename='feasible_dynamics.csv')

    if(collect_infeasible_dynamics):
        print('Collecting infeasible data:')
        for noise in [0.4, 0.6, 0.8, 1.0]:
            noisy_policy = NoisyPolicy(policy, 0.2) 
            collect_infeasible_data(env=env, policy=noisy_policy, noise=noise, eps=50, ep_len=450, state_dim=4, filename='infeasible_dynamics.csv')