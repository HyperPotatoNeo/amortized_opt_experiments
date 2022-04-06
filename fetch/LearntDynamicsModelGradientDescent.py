import argparse

import gym
import numpy as np
import torch

import sys
sys.path.append('../')

from learn_dynamics import MLP_Dynamics

from configs import observation_dim, action_dim, hidden_size, num_hidden
torch.set_printoptions(sci_mode=False)


class LearntModelGradientDescentPolicy:

    def __init__(self, learnt_model, x_mean, x_std, y_mean, y_std, action_dim, iters=20, lr=0.01, T=70):
        if isinstance(learnt_model, str):
            self.dynamics_model = MLP_Dynamics()
            self.dynamics_model.load_state_dict(torch.load(learnt_model).state_dict())
        elif isinstance(learnt_model, MLP_Dynamics):
            self.dynamics_model = learnt_model
        else:
            raise NotImplementedError
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.action_dim = action_dim
        self.iters = iters
        self.lr = lr
        self.T = T
        self.actions = torch.tensor(data=torch.FloatTensor(self.action_dim, self.T).uniform_(-1.0, 1.0),
                                    device=torch.device('cuda:0'), requires_grad=True)

    def __call__(self, obs, warm_start=False):
        if warm_start:
            self.actions = torch.nn.Parameter(
                data=torch.concat((self.actions[:, 1:], torch.zeros((self.action_dim, 1), device=torch.device('cuda:0'))), dim=1),
                requires_grad=True)
        else:
            self.actions = torch.tensor(data=torch.FloatTensor(self.action_dim, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cuda:0'), requires_grad=True)

        goal = torch.tensor(obs['desired_goal'], device='cuda:0')
        obs = torch.tensor(obs['observation'], device='cuda:0')

        optimizer = torch.optim.Adam({self.actions}, lr=self.lr)

        for i in range(self.iters):
            optimizer.zero_grad()
            current_state = obs

            rewards = 0
            for t in range(self.T):
                current_state = current_state + (self.dynamics_model(((torch.hstack([current_state, self.actions[:, t].reshape(self.action_dim)]).float() - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
                if current_state.shape[0] == 10:
                    reward = - ((current_state[0] - goal[0]) ** 2 + (current_state[1] - goal[1]) ** 2 + (current_state[2] - goal[2]) ** 2)
                else:
                    reward = - ((current_state[3] - goal[0]) ** 2 + (current_state[4] - goal[1]) ** 2 + (current_state[5] - goal[2]) ** 2)
                rewards -= reward
            rewards.backward()
            optimizer.step()

        return self.actions[:, 0].detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--filename', type=str, default='dynamics_data/FetchReach_data.npy', help='File name having data')
    parser.add_argument('--model-path', type=str, default='dynamics_models/FetchReach/dm_1.zip', help='Model path')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    obs = env.reset()

    data = np.load(args.filename)
    train_x = data[:, :observation_dim(args.env_name) + action_dim(args.env_name)]
    train_y = data[:, observation_dim(args.env_name) + action_dim(args.env_name):]

    x_mean = torch.tensor(train_x.mean(axis=0), device=torch.device('cuda:0'))
    x_std = torch.tensor(train_x.std(axis=0), device=torch.device('cuda:0'))
    ltrain_y = train_y - train_x[:, :observation_dim(args.env_name)]
    y_mean = torch.tensor(ltrain_y.mean(axis=0), device=torch.device('cuda:0'))
    y_std = torch.tensor(ltrain_y.std(axis=0), device=torch.device('cuda:0'))

    model = MLP_Dynamics(observation_dim(args.env_name),
                         action_dim(args.env_name),
                         hidden_size(args.env_name),
                         num_hidden(args.env_name)).cuda()
    model.load_state_dict(torch.load(args.model_path).state_dict())

    done = False

    policy = LearntModelGradientDescentPolicy(model, x_mean, x_std, y_mean, y_std, iters=20, action_dim=action_dim(args.env_name))

    while not done:
        obs, reward, done, _ = env.step(policy(obs, warm_start=True))
        env.render()
