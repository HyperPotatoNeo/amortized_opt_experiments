# coding: utf-8
import time
import pandas as pd
import numpy as np
import gym
import torch
import modified_gym_cartpole_swingup
torch.set_printoptions(sci_mode=False)

class MLP_direct_policy_model(torch.nn.Module):
    def __init__(self, state_dim=4):
        super(MLP_direct_policy_model, self).__init__()

        self.linear1 = torch.nn.Linear(state_dim*2, 256)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256,1)
        self.linear3 = torch.nn.Linear(256, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class FeasibilityCollocationPolicy:

    def __init__(self, f_model, T=70, iters=10, lr=0.01, N=1):
        self.T = T
        self.iters = iters
        self.lr = lr
        self.N = N
        self.rho = 2000
        self.mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
        self.mpc_env.mpc_reset(self.N * self.T)
        self.states = torch.tensor(data=torch.zeros(self.N * (self.T + 1), 4).float(),
                                   device=torch.device('cuda:0'),
                                   requires_grad=True)
        self.actions = torch.tensor(data=torch.FloatTensor(self.N * self.T).uniform_(-1.0, 1.0),
                                    device=torch.device('cuda:0'),
                                    requires_grad=True)
        self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                    device=torch.device('cuda:0'))

        indices = torch.arange(self.N * (self.T + 1))
        self.initial_indices = indices.reshape(self.N, self.T + 1)[:, :-1].reshape(self.N * self.T)
        self.final_indices = indices.reshape(self.N, self.T + 1)[:, 1:].reshape(self.N * self.T)
        self.f_model = f_model
        self.p = 0.99
        self.statistics = torch.tensor(pd.read_csv('feasibility_data/statistics.csv', header=None).values).float().cuda()


    def get_actions_from_states(self, iters = 500):
        self.actions.data *= 0
        action_optimizer = torch.optim.Adam({self.actions}, lr=0.01)

        for i in range(iters):
            action_optimizer.zero_grad()
            self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
            self.mpc_env.step(self.actions.reshape(self.N * self.T, 1))
            loss = torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2)
            loss.backward()
            action_optimizer.step()
            self.actions.data = torch.clamp(self.actions, -1, 1)
        #print((self.states[self.final_indices] - self.mpc_env.state) ** 2)
        #print('ACTION LOSS = ', loss)


    def __call__(self, state, warm_start=False, skip=1):
        if warm_start:
            self.mpc_env.mpc_reset(state=state)
            self.states.data[::self.T + 1] = self.mpc_env.state
            for t in range(self.T):
                self.mpc_env.step(self.actions[t::self.T].reshape(self.N, 1))
                self.states.data[t + 1::self.T + 1] = self.mpc_env.state

            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float() * 500,
                                        device=torch.device('cuda:0'))
        else:
            self.states.data *= 0.0
            self.states.data[::self.T + 1] = state
            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                        device=torch.device('cuda:0'))

        for i in range(self.iters):
            state_optimizer = torch.optim.Adam({self.states}, lr=0.01)

            for j in range(500):
                state_optimizer.zero_grad()
                self.states.data[::self.T + 1] = state
                self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
                rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N * self.T)
                rewards -= self.states[self.final_indices, 2].cos() - abs(self.states[self.final_indices, 0])
                state_transition_tensor = torch.cat((self.states[self.final_indices-1], self.states[self.final_indices]), dim=1)
                normalized_state_transition_tensor = (state_transition_tensor)*self.statistics[1]+self.statistics[0]
                dynamics_pred = torch.sigmoid(self.f_model(state_transition_tensor))[:,0]
                
                rewards += self.lambdas * (self.p - dynamics_pred)
                rewards += 0.5 * self.rho * ((self.p - dynamics_pred) ** 2)

                rewards = torch.sum(rewards)
                
                rewards.backward()
                state_optimizer.step()

            self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
            rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N * self.T)
            rewards -= self.states[self.final_indices, 2].cos() - abs(self.states[self.final_indices, 0])
            print(i, -torch.sum(rewards))
            #print(self.lambdas)
            self.states.data[::self.T + 1] = state
            self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
            self.mpc_env.step(self.actions.reshape(self.N * self.T, 1))
            self.lambdas.data += 0.5 * torch.log(self.p/(dynamics_pred + 0.01)) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000000.0)

        self.get_actions_from_states()
        print(self.actions)
        return self.states, self.actions


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 1
    T = 70
    f_model = torch.load('feasibility_data/feasibility_model')
    env.mpc_reset(N)

    done = False

    policy = FeasibilityCollocationPolicy(f_model, T=T, iters=40, N=N)

    for i in range(500):
        states, actions = policy(env.state.detach(), warm_start=True, skip=10)
        for j in range(10):
            env.step(actions[j::T].reshape(N, 1))
            env.render()
