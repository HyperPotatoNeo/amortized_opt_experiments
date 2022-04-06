import gym
import numpy as np
import os.path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyPolicy:
    def __init__(self, policy_model, action_noise_std=0.3):
        self.policy_model = policy_model
        self.action_noise_std = action_noise_std

    def __call__(self, obs):
        pi = self.policy_model(obs)
        res += np.random.normal(loc=0.0, scale=self.action_noise_std, size=1)[0]
        return res


class Actor(nn.Module):
    def __init__(self, env_params):
        super(Actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, env_params):
        super(Critic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--model-path', type=str, default='policy_models/FetchReach-v1/model.pt', help='Model path')
    parser.add_argument('--filename', type=str, default='FetchReach_data', help='File name for saving data')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to collect data')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')

    args = parser.parse_args()

    o_mean, o_std, g_mean, g_std, model = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()

    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = Actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()

    observation_shape = observation['observation'].shape[0]

    data_shape = env.action_space.shape[0] + 2 * observation_shape
    data = np.zeros((args.episodes * env._max_episode_steps, data_shape))

    for i in range(args.episodes):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            data[t + i * env._max_episode_steps, :observation_shape] = obs

            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            data[t + i * env._max_episode_steps, observation_shape:observation_shape + env.action_space.shape[0]] = action
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            data[t + i * env._max_episode_steps, observation_shape + env.action_space.shape[0]:] = obs

    if os.path.exists('dynamics_data/' + args.filename):
        prev_data = np.load('dynamics_data/' + args.filename, allow_pickle=True)
        data = np.vstack((prev_data, data))

    np.save('dynamics_data/' + args.filename, data)



