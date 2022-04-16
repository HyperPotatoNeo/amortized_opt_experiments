import gym
import numpy as np
import torch
import modified_gym_cartpole_swingup
from rewards_model import MLP_Dynamics
torch.set_printoptions(sci_mode=False)


class BatchCollocationLearntRewardPolicy:

    def __init__(self,  T=70, iters=10, lr=0.01, N=1):
        self.T = T
        self.iters = iters
        self.lr = lr
        self.N = N
        self.rho = 0
        self.epsilon = 1e-5
        self.mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
        self.mpc_env.reset()
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

        self.rewards_model = torch.load('rewards_models/rm_1.zip').cuda()
        data = np.load('dynamics_data/data_T_15_dagger.npy')[:, :4]
        self.x_mean = torch.tensor(data.mean(axis=0), device='cuda:0')
        self.x_std = torch.tensor(data.std(axis=0), device='cuda:0')

    def __call__(self, state, warm_start=False, skip=1):
        if warm_start:
            self.actions.data[:-skip] = self.actions.clone().data[skip:]

            self.mpc_env.mpc_reset(state=state)
            self.states.data *= 0.0
            self.states.data[::self.T + 1] = self.mpc_env.state
            for t in range(self.T):
                self.mpc_env.step(self.actions[t::self.T].reshape(self.N, 1))
                self.states.data[t + 1::self.T + 1] = self.mpc_env.state

            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                        device=torch.device('cuda:0'))
        else:
            self.states.data *= 0.0
            self.states.data[::self.T + 1] = state
            self.actions.data = torch.tensor(data=torch.FloatTensor(self.N * self.T).uniform_(-1.0, 1.0),
                                             device=torch.device('cuda:0'), requires_grad=True)
            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                        device=torch.device('cuda:0'))

        for i in range(self.iters):
            state_optimizer = torch.optim.Adam({self.states}, lr=0.1)
            action_optimizer = torch.optim.Adam({self.actions}, lr=0.05)

            for j in range(500):
                state_optimizer.zero_grad()
                action_optimizer.zero_grad()
                self.states.data[::self.T + 1] = state
                self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
                self.mpc_env.step(self.actions.reshape(self.N * self.T, 1))
                rewards = abs(self.states[self.final_indices, 0])
                rewards -= self.rewards_model(((self.states[self.final_indices] - self.x_mean) / self.x_std).float()).squeeze(axis=1)
                rewards += self.lambdas * (torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2, dim=1) - self.epsilon)
                rewards += 0.5 * self.rho * (torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2, dim=1) ** 2)

                rewards = torch.sum(rewards)
                rewards.backward()

                state_optimizer.step()
                action_optimizer.step()

                with torch.no_grad():
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)

            print(self.states)
            print((self.actions))

            self.states.data[::self.T + 1] = state
            self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
            self.mpc_env.step(self.actions.reshape(self.N * self.T, 1))
            self.lambdas.data += 0.1 * torch.log(torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2, dim=1) / self.epsilon + 0.01) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000000.0)

        return self.states, self.actions


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 1
    T = 70
    env.reset()
    env.mpc_reset(N)

    done = False

    policy = BatchCollocationLearntRewardPolicy(T=T, iters=30, N=N)

    for i in range(500):
        states, actions = policy(env.state.detach(), warm_start=True, skip=10)
        for j in range(10):
            env.step(actions[j::T].reshape(N, 1))
            env.render()
