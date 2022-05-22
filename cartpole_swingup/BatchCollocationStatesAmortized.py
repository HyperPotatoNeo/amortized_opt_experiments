# coding: utf-8
import random
import sys

import gym
import numpy as np
import torch
import modified_gym_cartpole_swingup
torch.set_printoptions(sci_mode=False, threshold=sys.maxsize)

from action_optimizer import ActionOptimizer


class BatchCollocationStatesAmortizedPolicy:

    def __init__(self,  T=70, iters=10, lr=0.01, N=1):
        self.T = T
        self.iters = iters
        self.lr = lr
        self.N = N
        self.rho = 200
        self.epsilon = 1e-5
        self.mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
        self.mpc_env.reset()
        self.mpc_env.mpc_reset(self.N * self.T)
        self.states = torch.tensor(data=torch.zeros(self.N * (self.T + 1), 4).float(),
                                   device=torch.device('cpu:0'),
                                   requires_grad=True)
        self.actions = torch.tensor(data=torch.FloatTensor(self.N * self.T).uniform_(-1.0, 1.0),
                                    device=torch.device('cpu:0'),
                                    requires_grad=True)
        self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                    device=torch.device('cpu:0'))

        indices = torch.arange(self.N * (self.T + 1))
        self.initial_indices = indices.reshape(self.N, self.T + 1)[:, :-1].reshape(self.N * self.T)
        self.final_indices = indices.reshape(self.N, self.T + 1)[:, 1:].reshape(self.N * self.T)

        self.action_optimizer_model = torch.load('action_models/data_collocation_optimal_continuous_cleaned_longer.zip', map_location='cuda')
        self.action_optimizer_model.eval()

        actions = torch.arange(-1.0, 1.025, 0.025)
        self.num_actions = len(actions)
        self.possible_actions = actions.repeat(self.N * self.T,).reshape(-1, 1)

        self.data = np.zeros((100000000, 8))
        self.index = 0

        actions = torch.arange(-0.25, + 0.25, 0.025)
        self.num_samples = len(actions)
        self.possible_samples = actions.repeat(self.N * self.T,).reshape(-1, 1)

    def __call__(self, state, warm_start=False, skip=1):
        if warm_start:
            self.actions.data[:-skip] = self.actions.clone().data[skip:]

            self.mpc_env.mpc_reset(state=state)
            self.states.data[::self.T + 1] = self.mpc_env.state
            for t in range(self.T):
                self.mpc_env.step(self.actions[t::self.T].reshape(self.N, 1))
                self.states.data[t + 1::self.T + 1] = self.mpc_env.state

            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                        device=torch.device('cpu:0'))
        else:
            self.states.data *= 0.0
            self.states.data[::self.T + 1] = state
            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                        device=torch.device('cpu:0'))

        for i in range(self.iters):
            state_optimizer = torch.optim.Adam({self.states}, lr=0.25)
            for j in range(400):
                if j != 0 and j % 200 == 0:
                    state_optimizer.param_groups[0]['lr'] = 0.2
                state_optimizer.zero_grad()
                self.states.data[::self.T + 1] = state
                self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
                self.mpc_env.step(self.actions.reshape(self.N * self.T, 1))
                rewards = torch.tensor(0.0, device=torch.device('cpu:0')).repeat(self.N * self.T)
                rewards -= self.states[self.final_indices, 2].cos() - abs(self.states[self.final_indices, 0])

                rewards += self.lambdas * (torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2, dim=1) - self.epsilon)
                rewards += 0.5 * self.rho * (torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2, dim=1) ** 2)

                rewards = torch.sum(rewards)
                rewards.backward()
                self.get_actions_from_states(i)
                state_optimizer.step()

            self.states.data[::self.T + 1] = state
            self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
            self.mpc_env.step(self.actions.reshape(self.N * self.T, 1))
            self.lambdas.data += 0.1 * torch.log(torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2, dim=1) / self.epsilon + 0.01) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000000.0)

        return self.states, self.actions

    def get_actions_from_states(self, i):
        # if random.random() < 0.01:
        #     self.data[self.index:self.index + len(self.states.data[self.initial_indices])] = np.hstack((self.states.data[self.initial_indices], self.states.data[self.final_indices]))
        #     self.index += len(self.states.data[self.initial_indices])

        if i < 20:
            x = torch.hstack((self.states.data[self.initial_indices], self.states.data[self.final_indices] - self.states.data[self.initial_indices])).float().cuda()

            self.actions.data = torch.clamp(self.action_optimizer_model(x).cpu(), -1.0, 1.0).reshape(-1)

            possible_actions = self.actions.data.repeat_interleave(self.num_samples,).reshape(-1, 1) + self.possible_samples
            possible_actions = torch.clamp(possible_actions, -1.0, 1.0)
            initial_states = self.states[self.initial_indices].repeat_interleave(self.num_samples, axis=0)
            final_states = self.states[self.final_indices].repeat_interleave(self.num_samples, axis=0)
            self.mpc_env.mpc_reset(state=initial_states)
            self.mpc_env.step(possible_actions)
            loss = torch.sum((final_states - self.mpc_env.state) ** 2, dim=1)
            self.actions.data = possible_actions.reshape(self.N * self.T, self.num_samples)[torch.vstack((torch.arange(0, self.N * self.T), loss.reshape(self.N * self.T, self.num_samples).argmin(axis=1))).T.numpy().T]

        else:
            initial_states = self.states[self.initial_indices].repeat_interleave(self.num_actions, axis=0)
            final_states = self.states[self.final_indices].repeat_interleave(self.num_actions, axis=0)
            self.mpc_env.mpc_reset(state=initial_states)
            self.mpc_env.step(self.possible_actions)
            loss = torch.sum((final_states - self.mpc_env.state) ** 2, dim=1)
            self.actions.data = self.possible_actions[loss.reshape(self.N * self.T, self.num_actions).argmin(axis=1)].reshape(-1)


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 20
    T = 70
    env.reset()
    env.mpc_reset(N)

    done = False

    policy = BatchCollocationStatesAmortizedPolicy(T=T, iters=25, N=N)

    for i in range(500):
        states, actions = policy(env.state.detach(), warm_start=True, skip=1)
        for j in range(1):
            env.step(policy.actions[j::T].reshape(N, 1))
            env.render()
            print(env.state)

    policy.data = policy.data[:policy.index]
    # np.save('dynamics_data/collocation_optimal.npy', policy.data)