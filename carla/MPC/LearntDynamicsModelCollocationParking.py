from car_env_for_MPC import CarEnv
from DynamicsModel import DynamicsModel
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
import matplotlib.pyplot as plt


class LearntModelCollocationPolicy:

    def __init__(self, learnt_model, T=45, iters=50, lr=0.005):
        self.dynamics_model = learnt_model
        self.T = T
        self.iters = iters
        self.lr = lr
        self.observation_dim = 5
        self.action_dim = 3
        self.epsilon = 1e-6
        self.rho = 200

        self.states = torch.tensor(data=torch.zeros(self.T + 1, self.observation_dim).float(),
                                   device=torch.device('cpu'), requires_grad=True)
        self.actions = torch.tensor(data=torch.FloatTensor(self.T, self.action_dim).uniform_(-1.0, 1.0),
                                    device=torch.device('cpu'), requires_grad=True)
        self.lambdas = torch.tensor(data=torch.ones(self.T).float(),
                                    device=torch.device('cpu'))

        self.initial_indices = torch.arange(self.T)
        self.final_indices = torch.arange(1, self.T + 1)

    def __call__(self, obs, waypoints, warm_start=False, skip=1, call=1):

        obs = torch.tensor(obs, device='cpu')
        waypoints = torch.tensor(waypoints, device='cpu')

        if warm_start:
            self.T -= skip
            self.T = max(self.T, 5)
            previous = self.actions.clone().data[skip:, :]

            self.actions = torch.tensor(data=torch.FloatTensor(self.T, self.action_dim).uniform_(-1.0, 1.0),
                                        device=torch.device('cpu'), requires_grad=True)
            self.actions.data[:len(previous), :] = previous

            self.states = torch.tensor(data=torch.zeros(self.T + 1, self.observation_dim).float(),
                                       device=torch.device('cpu'), requires_grad=True)
            self.states.data[0, :] = obs
            for t in range(self.T):
                self.states.data[t + 1, :] = self.dynamics_model.discrete_dynamics(self.states[t, :], self.actions[t, :])

            self.lambdas = torch.tensor(data=torch.ones(self.T).float(),
                                        device=torch.device('cpu'))

            self.initial_indices = torch.arange(self.T)
            self.final_indices = torch.arange(1, self.T + 1)

        else:
            self.actions = torch.tensor(data=torch.FloatTensor(self.action_dim, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cpu'), requires_grad=True)

        state_optimizer = torch.optim.Adam({self.states}, lr=self.lr)
        action_optimizer = torch.optim.Adam({self.actions}, lr=self.lr)

        for i in range(self.iters):
            for j in range(500):
                state_optimizer.zero_grad()
                action_optimizer.zero_grad()

                self.states.data[0] = obs

                next_states = self.dynamics_model.discrete_dynamics(self.states[self.initial_indices], self.actions)
                rewards = torch.tensor(0.0, device=torch.device('cpu')).repeat(self.T)
                rewards += self.lambdas * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) - self.epsilon)
                rewards += 0.5 * self.rho * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) ** 2)

                rewards[-8:] += 500 * (self.states[-min(8, len(rewards)):, 3] - 3.14) ** 2

                rewards[-1] += 50 * ((self.states[-1, 0] - waypoints[0]) ** 2 + 2 * (self.states[-1, 1] - waypoints[1]) ** 2)
                rewards[-1] += 50 * (self.states[-1, 2]) ** 2

                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -41.16636276245117) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -38.57696533203125) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -35.526119232177734) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -32.72093963623047) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -30.194425582885742) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -21.28460693359375) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -18.913570404052734) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -32.89897537231445) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -35.526119232177734) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -30.194425582885742) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -18.60922622680664) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -21.28460693359375) ** 2 / 16, 0, 1))
                rewards -= 16 * (torch.clip((self.states[self.final_indices, 0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((self.states[self.final_indices, 1] - -15.445639610290527) ** 2 / 16, 0, 1))

                rewards = torch.sum(rewards)
                rewards.backward()
                torch.nn.utils.clip_grad_value_(self.states, 100.0)

                state_optimizer.step()
                action_optimizer.step()
                self.states.data[0] = obs

                with torch.no_grad():
                    self.states.data[:, 2] = torch.clamp(self.states[:, 2], 0.0, 20.0)
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)
                    self.actions.data[:, 1:] = torch.clamp(self.actions[:, 1:], 0.0, 1.0)

                # if j % 200 == 0:
                #     plt.scatter(obs[1], obs[0], color='red')
                #     plt.scatter(self.states[:, 1].data, self.states[:, 0].data, color='blue')
                #     plt.scatter(target[1], target[0], color='green')
                #     plt.scatter(-41.16636276245117, -28.6, color='orange')
                #     plt.scatter(-38.57696533203125, -28.6, color='orange')
                #     plt.scatter(-35.526119232177734, -22.6, color='orange')
                #     plt.scatter(-32.72093963623047, -22.6, color='orange')
                #     plt.scatter(-30.194425582885742, -22.6, color='orange')
                #     plt.scatter(-21.28460693359375, -28.6, color='orange')
                #     plt.scatter(-18.913570404052734, -28.6, color='orange')
                #     plt.scatter(-30.194425582885742, -12.52372932434082, color='orange')
                #     plt.scatter(-32.89897537231445, -12.52372932434082, color='orange')
                #     plt.scatter(-35.526119232177734, -12.52372932434082, color='orange')
                #     plt.scatter(-18.60922622680664, -12.52372932434082, color='orange')
                #     plt.gca().set_aspect('equal', adjustable='box')
                #     plt.show()

            self.states.data[0] = obs
            next_states = self.dynamics_model.discrete_dynamics(self.states[self.initial_indices], self.actions)
            self.lambdas.data += 0.05 * torch.log(torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) / self.epsilon + 0.01) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000000.0)

        actual_states = torch.zeros((self.T + 1, 5), requires_grad=False)
        actual_states[0, :] = obs
        for t in range(self.T):
            actual_states[t + 1] = self.dynamics_model.discrete_dynamics(actual_states[t], self.actions[t])

        return self.states[:, :2].detach().numpy(), actual_states[:, :2].detach().numpy(), self.actions


if __name__ == "__main__":
    env = CarEnv()
    state, waypoints = env.reset(parking=True)

    model = DynamicsModel()

    done = False

    policy = LearntModelCollocationPolicy(model)

    target = [-22.465097427368164, -18.80262565612793]
    skip = 1
    try:
        for i in range(500):
            states, actual_states, actions = policy(state, target, skip=skip, warm_start=True, call=i+1)

            for j in range(min(skip, len(actions))):
                state, waypoints, done, _ = env.step(actions[j], states, actual_states)

            skip = min(5, len(actions))

    finally:
        print('destroying actors')
        for actor in env.actor_list:
            actor.destroy()