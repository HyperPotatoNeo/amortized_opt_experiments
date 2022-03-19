import gym
import gym_cartpole_swingup
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os.path


class NoisyPolicy:
    def __init__(self, policy_model, action_noise_std=0.3):
        self.policy_model = policy_model
        self.action_noise_std = action_noise_std

    def __call__(self, obs):
        res, _states = self.policy_model.predict(obs)
        res += np.random.normal(loc=0.0, scale=self.action_noise_std, size=1)[0]
        return res


def learn_policy_model(env, timesteps, name):
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(timesteps)
    model.save('policy_models/'+name)

    for r in range(10):
        obs = env.reset()
        for _ in range(40000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()


def collect_dynamics_data(env, policy_model, eps=2, ep_len=400, state_dim=4, action_dim=1, filename='data.npy'):
    data = np.zeros((eps,ep_len,state_dim+action_dim+state_dim))

    for i in range(eps):
        print(i)
        obs = env.reset()
        for j in range(ep_len):
            data[i,j,:state_dim] = env.state
            action = policy_model(obs)
            data[i,j,state_dim:(state_dim+action_dim)] = action
            obs, rewards, dones, info = env.step(action)
            data[i,j,(state_dim+action_dim):] = env.state
            if(i%10==0):
                env.render()

    if(os.path.exists('dynamics_data/'+filename)):
        data_prev = np.load('dynamics_data/'+filename, allow_pickle=True)
        data = np.vstack((data_prev,data))

    np.save(filename, data)

if __name__=='__main__':
    env = gym.make("CartPoleSwingUp-v0")
    learn_policy = False
    learn_dynamics = True
    collect_data = True
    T = 15

    if(learn_policy):
        policy_model = learn_policy_model(env, 100000, 'mlp_1')

    else:
        policy_model = PPO.load('policy_models/mlp_1')

    noisy_policy_model = NoisyPolicy(policy_model, 0.4)

    if(collect_data):
        collect_dynamics_data(env=env, policy_model=noisy_policy_model, eps=400, ep_len=300, state_dim=4, action_dim=1, filename='data_T_15.npy')






