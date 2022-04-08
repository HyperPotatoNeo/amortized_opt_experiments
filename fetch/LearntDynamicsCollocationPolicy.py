# coding: utf-8
import argparse

import gym
import numpy as np
import torch

from learn_dynamics import MLP_Dynamics, LearnDynamics

torch.set_printoptions(sci_mode=False)


class LearntDynamicsCollocationPolicy:

    def __init__(self, learnt_model, x_mean, x_std, y_mean, y_std, T=70, iters=10, lr=0.01, observation_dim=None, action_dim=None):
        self.dynamics_model = learnt_model
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.T = T
        self.iters = iters
        self.lr = lr
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.rho = 200
        self.epsilon = 1e-2
        self.states = torch.tensor(data=torch.FloatTensor(self.T + 1, self.observation_dim).uniform_(-1.0, 1.0),
                                   device=torch.device('cuda:0'),
                                   requires_grad=True)
        self.actions = torch.tensor(data=torch.FloatTensor(self.T, self.action_dim).uniform_(-1.0, 1.0),
                                    device=torch.device('cuda:0'),
                                    requires_grad=True)
        self.lambdas = torch.tensor(data=torch.ones(self.T).float(),
                                    device=torch.device('cuda:0'))
        self.initial_indices = torch.arange(self.T)
        self.final_indices = torch.arange(1, self.T + 1)

    def __call__(self, obs, warm_start=False, skip=1):
        goal = torch.tensor(obs['desired_goal'], device='cuda:0')
        obs = torch.tensor(obs['observation'], device='cuda:0')

        self.states.data[:] = obs
        if warm_start:
            self.actions.data[:-skip, :] = self.actions.clone().data[skip:, :]

            self.lambdas = torch.tensor(data=torch.ones(self.T).float() * 500,
                                        device=torch.device('cuda:0'))
        else:
            self.states.data *= 0.0
            self.states.data[0] = obs
            self.actions.data = torch.tensor(data=torch.FloatTensor(self.T).uniform_(-1.0, 1.0),
                                             device=torch.device('cuda:0'), requires_grad=True)
            self.lambdas = torch.tensor(data=torch.ones(self.T).float(),
                                        device=torch.device('cuda:0'))

        for i in range(self.iters):
            state_optimizer = torch.optim.Adam({self.states}, lr=0.005)
            action_optimizer = torch.optim.Adam({self.actions}, lr=0.01)

            for j in range(500):
                state_optimizer.zero_grad()
                action_optimizer.zero_grad()
                self.states.data[0] = obs
                next_states = self.states[self.initial_indices] + (self.dynamics_model(((torch.hstack([self.states[self.initial_indices], self.actions]) - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
                rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.T)
                if next_states.shape[1] == 10:
                    rewards += ((next_states[:, 0] - goal[0]) ** 2 + (next_states[:, 1] - goal[1]) ** 2 + (next_states[:, 2] - goal[2]) ** 2)
                else:
                    rewards += ((next_states[:, 3] - goal[0]) ** 2 + (next_states[:, 4] - goal[1]) ** 2 + (next_states[:, 5] - goal[2]) ** 2)
                    # rewards += ((next_states[:, 0] - goal[0]) ** 2 + (next_states[:, 1] - goal[1]) ** 2 + (next_states[:, 2] - goal[2]) ** 2)
                rewards += self.lambdas * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) - self.epsilon)
                rewards += 0.5 * self.rho * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) ** 2)
                rewards = torch.sum(rewards)
                rewards.backward()
                state_optimizer.step()
                action_optimizer.step()

                with torch.no_grad():
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)

            self.states.data[0] = obs
            next_states = self.states[self.initial_indices] + (self.dynamics_model(((torch.hstack([self.states[self.initial_indices], self.actions]) - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
            self.lambdas.data += 0.1 * torch.log(torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) / self.epsilon + 0.01) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000000.0)

        return self.states, self.actions.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--filename', type=str, default='dynamics_data/FetchReach_data.npy', help='File name having data')
    parser.add_argument('--model-path', type=str, default='dynamics_models/FetchReach/dm_1.zip', help='Model path')
    parser.add_argument('--epochs', type=int, default=30000, help='Number of epochs for training')
    parser.add_argument('--decay-after', type=int, default=10000, help='Decay learning rate after')
    parser.add_argument('--decay-factor', type=float, default=10.0, help='Learning rate decay factor')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num-steps', type=int, default=0, help='Number of steps after which to learn dynamics')

    args = parser.parse_args()

    env = gym.make(args.env_name)
    obs = env.reset()

    data = np.load(args.filename)
    train_x = data[:, :obs['observation'].shape[0] + env.action_space.shape[0]]
    train_y = data[:, obs['observation'].shape[0] + env.action_space.shape[0]:]

    x_mean = torch.tensor(train_x.mean(axis=0), device=torch.device('cuda:0'))
    x_std = torch.tensor(train_x.std(axis=0), device=torch.device('cuda:0'))
    ltrain_y = train_y - train_x[:, :obs['observation'].shape[0]]
    y_mean = torch.tensor(ltrain_y.mean(axis=0), device=torch.device('cuda:0'))
    y_std = torch.tensor(ltrain_y.std(axis=0), device=torch.device('cuda:0'))

    model = torch.load(args.model_path)

    done = False

    if args.num_steps:
        learn_dynamics = LearnDynamics(observation_dim=obs['observation'].shape[0],
                                       action_dim=env.action_space.shape[0],
                                       filename=args.filename,
                                       learning_rate=args.learning_rate,
                                       epochs=args.epochs,
                                       decay_factor=args.decay_factor,
                                       decay_after=args.decay_after,
                                       model_path=args.model_path)
        data = np.zeros((args.num_steps, env.action_space.shape[0] + 2 * obs['observation'].shape[0]))
        index = 0

    policy = LearntDynamicsCollocationPolicy(model, x_mean, x_std, y_mean, y_std, observation_dim=obs['observation'].shape[0], action_dim=env.action_space.shape[0], iters=30, T=10)

    for _ in range(500):
        obs = env.reset()
        done = False
        while not done:
            _, actions = policy(obs, warm_start=True, skip=1)

            if args.num_steps:
                data[index, :obs['observation'].shape[0]] = obs['observation']
                data[index, obs['observation'].shape[0]:obs['observation'].shape[0] + env.action_space.shape[0]] = actions[0]

            obs, reward, done, _ = env.step(actions[0])

            if args.num_steps:
                data[index, obs['observation'].shape[0] + env.action_space.shape[0]:] = obs['observation']
                index = (index + 1) % args.num_steps
                if index == 0:
                    learn_dynamics.append_data(data)
                    np.save(args.filename, learn_dynamics.data)
                    policy.model = learn_dynamics.learn_dynamics()
                    index = 0

            env.render()

