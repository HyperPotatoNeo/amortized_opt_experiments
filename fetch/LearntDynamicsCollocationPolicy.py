# coding: utf-8
import argparse

import gym
import numpy as np
import torch

from learn_dynamics import MLP_Dynamics, LearnDynamics

torch.set_printoptions(sci_mode=False, profile="full", linewidth=2000)

import joblib
from networks import OneStepModelFC
from one_step_model import SimpleOneStepModel
from envs.fetch_push import FetchPush


class LearntDynamicsCollocationPolicy:

    def __init__(self, learnt_model, x_mean, x_std, y_mean, y_std, T=70, iters=10, lr=0.01, observation_dim=None, action_dim=None):
        self.dynamics_model = learnt_model
        self.dynamics_model.networks[0].eval()
        self.x_mean = torch.tensor(self.dynamics_model.networks[0].state_scaler.mean_, device='cuda:0').float()
        self.x_std = torch.tensor(self.dynamics_model.networks[0].state_scaler.scale_, device='cuda:0').float()
        self.y_mean = torch.tensor(self.dynamics_model.networks[0].diff_scaler.mean_, device='cuda:0').float()
        self.y_std = torch.tensor(self.dynamics_model.networks[0].diff_scaler.scale_, device='cuda:0').float()
        self.T = T
        self.iters = iters
        self.lr = lr
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.rho = 200
        self.epsilon = 1e-6
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

    def __call__(self, observation, warm_start=False, skip=1):
        goal = torch.tensor(observation['desired_goal'], device='cuda:0')
        obs = torch.tensor(observation['observation'], device='cuda:0')

        self.states.data[:] = obs
        if warm_start:
            self.actions.data[:-skip, :] = self.actions.clone().data[skip:, :]

            self.states.data *= 0.0
            self.states.data[0] = obs
            self.states.data[:, :3] = torch.tensor(observation['achieved_goal'], device='cuda:0')
            self.states.data[:, 3:6] = self.states.data[:, :3]
            # for t in range(self.T):
            #     self.states.data[t + 1] = self.states[t] + self.dynamics_model.networks[0]((self.states[t] - self.x_mean) / self.x_std, self.actions[t]) * self.y_std + self.y_mean

            # self.states.data[:-skip, :] = self.states.clone().data[skip:, :]
            self.lambdas = torch.tensor(data=torch.ones(self.T).float() * 500,
                                        device=torch.device('cuda:0'))
        else:
            self.states.data *= 0.0
            self.states.data[0] = obs
            self.actions.data = torch.tensor(data=torch.FloatTensor(self.T).uniform_(-1.0, 1.0),
                                             device=torch.device('cuda:0'), requires_grad=True)
            self.lambdas = torch.tensor(data=torch.ones(self.T).float() * 500,
                                        device=torch.device('cuda:0'))

        state_optimizer = torch.optim.Adam({self.states}, lr=0.0005)
        action_optimizer = torch.optim.Adam({self.actions}, lr=0.05)
        for i in range(self.iters):
            for j in range(1500):
                state_optimizer.zero_grad()
                action_optimizer.zero_grad()
                self.states.data[0] = obs
                # next_states = self.states[self.initial_indices] + (self.dynamics_model(((torch.hstack([self.states[self.initial_indices], self.actions]) - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
                next_states = self.states[self.initial_indices] + self.dynamics_model.networks[0]((self.states[self.initial_indices] - self.x_mean) / self.x_std, self.actions) * self.y_std + self.y_mean
                rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.T)
                if next_states.shape[1] == 10:
                    rewards += ((next_states[:, 0] - goal[0]) ** 2 + (next_states[:, 1] - goal[1]) ** 2 + (next_states[:, 2] - goal[2]) ** 2)
                else:
                    rewards += ((self.states[self.final_indices, 3] - goal[0]) ** 2 + (self.states[self.final_indices, 4] - goal[1]) ** 2)
                    # rewards += ((self.states[self.final_indices, 3] - self.states[self.final_indices, 0]) ** 2 + (self.states[self.final_indices, 4] - self.states[self.final_indices, 1]) ** 2)
                    # rewards += (self.states[self.final_indices, 6] ** 2 + self.states[self.final_indices, 7] ** 2)
                    # rewards += ((next_states[:, 0] - goal[0]) ** 2 + (next_states[:, 1] - goal[1]) ** 2 + (next_states[:, 2] - goal[2]) ** 2)
                    # print(torch.sum(rewards))
                rewards += self.lambdas * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) - self.epsilon)
                rewards += 0.5 * self.rho * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) ** 2)
                rewards = torch.sum(rewards)
                rewards.backward()
                # print(torch.linalg.norm(self.states.grad))
                # torch.nn.utils.clip_grad_norm_(self.states, 100)
                # torch.nn.utils.clip_grad_value_(self.states, 100)
                # print(self.states.grad, self.actions.grad)
                state_optimizer.step()
                action_optimizer.step()

                with torch.no_grad():
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)

                    self.states.data[self.final_indices, 0] = torch.clip(self.states[self.final_indices, 0], 1.15, 1.55)
                    self.states.data[self.final_indices, 1] = torch.clip(self.states[self.final_indices, 1], 0.55, 0.95)
                    self.states.data[self.final_indices, 2] = torch.clip(self.states[self.final_indices, 2], 0.41, 0.5)
                    self.states.data[self.final_indices, 3] = torch.clip(self.states[self.final_indices, 3], 1.2, 1.5)
                    self.states.data[self.final_indices, 4] = torch.clip(self.states[self.final_indices, 4], 0.6, 0.9)
                    self.states.data[self.final_indices, 5] = torch.clip(self.states[self.final_indices, 5], 0.4, 0.47)

            print('expected', self.states[:, :9])

            self.states.data[0] = obs
            # next_states = self.states[self.initial_indices] + (self.dynamics_model(((torch.hstack([self.states[self.initial_indices], self.actions]) - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
            next_states = self.states[self.initial_indices] + self.dynamics_model.networks[0]((self.states[self.initial_indices] - self.x_mean) / self.x_std, self.actions) * self.y_std + self.y_mean
            print('constraint', torch.hstack(((torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1).reshape(-1, 1) < self.epsilon), self.lambdas.reshape(-1, 1))))
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

    # env = gym.make(args.env_name)
    env = FetchPush(remove_gripper=True)
    obs = env.reset()

    data = np.load(args.filename)
    train_x = data[:, :obs['observation'].shape[0] + env.action_space.shape[0]]
    train_y = data[:, obs['observation'].shape[0] + env.action_space.shape[0]:]

    x_mean = torch.tensor(train_x.mean(axis=0), device=torch.device('cuda:0'))
    x_std = torch.tensor(train_x.std(axis=0), device=torch.device('cuda:0'))
    # ltrain_y = train_y - train_x[:, :obs['observation'].shape[0]]
    y_mean = torch.tensor(train_y.mean(axis=0), device=torch.device('cuda:0'))
    y_std = torch.tensor(train_y.std(axis=0), device=torch.device('cuda:0'))

    # model = torch.load(args.model_path)
    _, osm_params = joblib.load('experiments/FetchPush/parameters_final.pkl')

    osms = [OneStepModelFC(24, 4, hidden_sizes=[512, 512], device="cuda:0").cuda(),
            OneStepModelFC(24, 4, hidden_sizes=[512, 512], device="cuda:0"),
            OneStepModelFC(24, 4, hidden_sizes=[512, 512], device="cuda:0")]

    model = SimpleOneStepModel(osms, [], l2_reg=0.0001, device="cuda")

    model.load(*osm_params)

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

    policy = LearntDynamicsCollocationPolicy(model, x_mean, x_std, y_mean, y_std, observation_dim=obs['observation'].shape[0], action_dim=env.action_space.shape[0], iters=30, T=20)

    skip = 5
    for _ in range(500):
        obs = env.reset()
        env.render()
        done = False
        print('actual', obs)
        while not done:
            states, actions = policy(obs, warm_start=True, skip=skip)
            print('expected', states[:, :9])
            print('expected', states[skip, :9])

            if args.num_steps:
                data[index, :obs['observation'].shape[0]] = obs['observation']
                data[index, obs['observation'].shape[0]:obs['observation'].shape[0] + env.action_space.shape[0]] = actions[0]

            for i in range(skip):
                print('actions', actions[i])
                obs, reward, done, _ = env.step(actions[i])
                env.render()

            print('actual', obs)
            if args.num_steps:
                data[index, obs['observation'].shape[0] + env.action_space.shape[0]:] = obs['observation']
                index = (index + 1) % args.num_steps
                if index == 0:
                    learn_dynamics.append_data(data)
                    np.save(args.filename, learn_dynamics.data)
                    policy.model = learn_dynamics.learn_dynamics()
                    index = 0


