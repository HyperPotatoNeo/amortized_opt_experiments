# coding: utf-8
import time

import gym
import torch
import modified_gym_cartpole_swingup
torch.set_printoptions(sci_mode=False)


class BatchCollocationPolicy:

    def __init__(self,  T=70, iters=10, lr=0.01, N=1):
        self.T = T
        self.iters = iters
        self.lr = lr
        self.N = N
        self.mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
        self.mpc_env.mpc_reset(self.N)
        self.states = torch.tensor(data=torch.zeros(self.N, 4, self.T).float(),
                                   device=torch.device('cuda:0'),
                                   requires_grad=True)
        self.actions = torch.tensor(data=torch.FloatTensor(self.N, self.T).uniform_(-1.0, 1.0),
                                    device=torch.device('cuda:0'),
                                    requires_grad=True)
        self.lambdas = torch.tensor(data=torch.ones(self.N, 4, self.T).float(),
                                    device=torch.device('cuda:0'),
                                    requires_grad=True)

    def __call__(self, state, warm_start=False):
        if warm_start:
            self.actions.data[:, :-1] = self.actions.clone().data[:, 1:]
            self.mpc_env.mpc_reset(state=state)
            for t in range(self.T):
                self.mpc_env.step(self.actions[:, t].reshape(self.N, 1))
                self.states.data[:, :, t] = self.mpc_env.state
            self.lambdas.data[:, :, :-1] = self.lambdas.clone().data[:, :, 1:]
        else:
            self.states.data *= 0.0
            self.actions = torch.tensor(data=torch.FloatTensor(self.N, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cuda:0'), requires_grad=True)
            self.lambdas *= 0.0

        optimizer = torch.optim.SGD({self.states, self.actions}, lr=0.01, momentum=0.9)
        lambda_optimizer = torch.optim.SGD({self.lambdas}, lr=0.1, momentum=0.9)

        for i in range(self.iters):
            for j in range(1):
                optimizer.zero_grad()
                self.mpc_env.mpc_reset(state=state)
                rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N, 4)
                for t in range(self.T):
                    self.mpc_env.step(self.actions[:, t].reshape(self.N, 1))
                    rewards -= self.states[:, 2, t].cos() - abs(self.states[:, 0, t])
                    rewards += self.lambdas[:, :, t] * ((self.states[:, :, t] - self.mpc_env.state) ** 2)
                    self.mpc_env.mpc_reset(state=self.states[:, :, t])
                rewards = torch.sum(rewards)
                rewards.backward()
                optimizer.step()

                with torch.no_grad():
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)

            lambda_optimizer.zero_grad()
            self.mpc_env.mpc_reset(state=state)
            rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N, 4)
            for t in range(self.T):
                self.mpc_env.step(self.actions[:, t].reshape(self.N, 1))
                rewards -= self.lambdas[:, :, t] * ((self.states[:, :, t] - self.mpc_env.state) ** 2)
                rewards += self.states[:, 2, t].cos() - abs(self.states[:, 0, t])
                self.mpc_env.mpc_reset(state=self.states[:, :, t])
            rewards = torch.sum(rewards)
            rewards.backward()
            lambda_optimizer.step()

            with torch.no_grad():
                self.lambdas.data = torch.clamp(self.lambdas, 0.0, 999999999999999)

        return self.states, self.actions


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 1
    env.mpc_reset(N)

    done = False

    policy = BatchCollocationPolicy(N=N)

    for i in range(500):
        _, actions = policy(env.state.detach(), warm_start=True)
        env.step(actions)
        if i % 5 == 0:
            env.render()
