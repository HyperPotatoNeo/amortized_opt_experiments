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


class LearntDynamicsCollocationSecondOrderPolicy:

    def __init__(self, learnt_model, x_mean, x_std, y_mean, y_std, T=70, iters=10, lr=1, observation_dim=None, action_dim=None):
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
        self.rho = 0
        self.epsilon = 1e-6
        self.damping = 1
        self.states = torch.tensor(data=torch.FloatTensor(self.T + 1, self.observation_dim).uniform_(-1.0, 1.0),
                                   device=torch.device('cuda:0'))
        self.actions = torch.tensor(data=torch.FloatTensor(self.T, self.action_dim).uniform_(-1.0, 1.0),
                                    device=torch.device('cuda:0'))
        self.lambdas = torch.tensor(data=torch.ones(self.T).float(),
                                    device=torch.device('cuda:0'))
        self.initial_indices = torch.arange(self.T)
        self.final_indices = torch.arange(1, self.T + 1)

    def optimize(self, J, residuals, variables):
        with torch.no_grad():
            JTJ = torch.t(J) @ J
            dx = torch.linalg.inv(JTJ + self.damping * torch.diag(torch.diag(JTJ))) @ torch.t(J) @ residuals
            variables[self.observation_dim:] = variables[self.observation_dim:] - self.lr * dx
            self.states.data = torch.reshape(variables[:(self.T + 1) * self.observation_dim], (self.T + 1, self.observation_dim))
            self.actions.data = torch.reshape(variables[(self.T + 1) * self.observation_dim:], (self.T, self.action_dim))

    def residuals(self, inputs):
        states = torch.reshape(inputs[:(self.T + 1) * self.observation_dim], (self.T + 1, self.observation_dim))
        actions = torch.reshape(inputs[(self.T + 1) * self.observation_dim:], (self.T, self.action_dim))

        next_states = states[self.initial_indices] + self.dynamics_model.networks[0](
            (states[self.initial_indices] - self.x_mean) / self.x_std, actions) * self.y_std + self.y_mean

        if next_states.shape[1] == 10:
            residuals = torch.cat((states[self.final_indices, 0] - self.goal[0],
                                   states[self.final_indices, 1] - self.goal[1],
                                   states[self.final_indices, 2] - self.goal[2]))
        else:
            residuals = torch.cat((states[self.final_indices, 3] - self.goal[0],
                                   states[self.final_indices, 4] - self.goal[1]))
            # residuals = torch.cat((residuals,
            #                        states[self.final_indices, 6],
            #                        states[self.final_indices, 7]))
            # residuals = torch.cat((residuals,
            #                        states[self.final_indices, 0] - states[self.final_indices, 3],
            #                        states[self.final_indices, 1] - states[self.final_indices, 4]))
        residuals = torch.cat((residuals, torch.sqrt(self.lambdas).tile(self.observation_dim) * (states[self.final_indices] - next_states).T.flatten()))
        self.current_residuals = residuals
        return residuals

    def __call__(self, observation, warm_start=False, skip=1, env=None):
        self.goal = torch.tensor(observation['desired_goal'], device='cuda:0')
        obs = torch.tensor(observation['observation'], device='cuda:0')

        self.states.data[:] = obs
        if warm_start:
            self.actions.data[:-skip, :] = self.actions.clone().data[skip:, :]

            # self.states.data *= 0.0
            # self.states.data[:, :3] = torch.tensor(observation['achieved_goal'], device='cuda:0')
            # self.states.data[:, 3:6] = self.states.data[:, :3]
            self.states.data[0] = obs
            for t in range(self.T):
                self.states.data[t + 1] = self.states[t] + self.dynamics_model.networks[0]((self.states[t] - self.x_mean) / self.x_std, self.actions[t]) * self.y_std + self.y_mean

            # self.states.data[:-skip, :] = self.states.clone().data[skip:, :]
            self.lambdas = torch.tensor(data=torch.ones(self.T).float() * 5,
                                        device=torch.device('cuda:0'))
        else:
            self.states.data *= 0.0
            self.states.data[0] = obs
            self.actions.data = torch.tensor(data=torch.FloatTensor(self.T).uniform_(-1.0, 1.0),
                                             device=torch.device('cuda:0'), requires_grad=True)
            self.lambdas = torch.tensor(data=torch.ones(self.T).float() * 500,
                                        device=torch.device('cuda:0'))


        i = 0
        while i < 50 and (i < self.iters or ~((torch.sum((self.states[1:6] - next_states[:5]) ** 2, dim=1) < self.epsilon).all())):
            i += 1
            for j in range(30):
                self.states.data[0] = obs

                variables = torch.cat([torch.flatten(self.states), torch.flatten(self.actions)])
                J = torch.autograd.functional.jacobian(self.residuals, variables)
                self.optimize(J[:, self.observation_dim:], self.current_residuals, variables)

                if j > 0:
                    if torch.sum(self.current_residuals ** 2) < torch.sum(self.previous_residuals ** 2):
                        self.damping = self.damping / 5
                    else:
                        self.damping = self.damping * 2
                self.damping = np.clip(self.damping, 0.001, None)
                self.previous_residuals = self.current_residuals

                with torch.no_grad():
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)

                    self.states.data[self.final_indices, 0] = torch.clip(self.states[self.final_indices, 0], 1.15, 1.55)
                    self.states.data[self.final_indices, 1] = torch.clip(self.states[self.final_indices, 1], 0.55, 0.95)
                    self.states.data[self.final_indices, 2] = torch.clip(self.states[self.final_indices, 2], 0.41, 0.5)
                    self.states.data[self.final_indices, 3] = torch.clip(self.states[self.final_indices, 3], 1.15, 1.55)
                    self.states.data[self.final_indices, 4] = torch.clip(self.states[self.final_indices, 4], 0.55, 0.95)
                    self.states.data[self.final_indices, 5] = torch.clip(self.states[self.final_indices, 5], 0.41, 0.5)

            print('expected', self.states[:, :9])

            self.states.data[0] = obs
            # next_states = self.states[self.initial_indices] + (self.dynamics_model(((torch.hstack([self.states[self.initial_indices], self.actions]) - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
            next_states = self.states[self.initial_indices] + self.dynamics_model.networks[0]((self.states[self.initial_indices] - self.x_mean) / self.x_std, self.actions) * self.y_std + self.y_mean
            print('constraint', torch.hstack(((torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1).reshape(-1, 1) < self.epsilon), self.lambdas.reshape(-1, 1))))
            self.lambdas.data += 0.1 * torch.log(torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) / self.epsilon + 0.01) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000000.0)
            env.render()

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

    policy = LearntDynamicsCollocationSecondOrderPolicy(model, x_mean, x_std, y_mean, y_std, observation_dim=obs['observation'].shape[0], action_dim=env.action_space.shape[0], iters=30, T=20)

    skip = 1
    for _ in range(500):
        obs = env.reset()
        env.render()
        done = False
        print('actual', obs)
        while not done:
            states, actions = policy(obs, warm_start=True, skip=skip, env=env)
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


