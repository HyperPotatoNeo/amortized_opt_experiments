from car_env_for_MPC import CarEnv
from DynamicsModel import DynamicsModel
from action_optimizer import ActionOptimizer
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)


class LearntModelCollocationAmortizedPolicy:

    def __init__(self, learnt_model, T=10, iters=30, lr=0.005):
        self.dynamics_model = learnt_model
        self.T = T
        self.iters = iters
        self.lr = lr
        self.observation_dim = 5
        self.action_dim = 3
        self.epsilon = 1e-5
        self.rho = 100

        self.states = torch.tensor(data=torch.zeros(self.T + 1, self.observation_dim).float(),
                                   device=torch.device('cpu'), requires_grad=True)
        self.actions = torch.tensor(data=torch.FloatTensor(self.T, self.action_dim).uniform_(-1.0, 1.0),
                                    device=torch.device('cpu'), requires_grad=True)
        self.lambdas = torch.tensor(data=torch.ones(self.T).float(),
                                    device=torch.device('cpu'))

        self.initial_indices = torch.arange(self.T)
        self.final_indices = torch.arange(1, self.T + 1)

        self.action_optimizer_model = torch.load('../SystemID/inverse_model/action_model', map_location='cuda')
        self.action_optimizer_model.eval()

    def __call__(self, obs, waypoints, warm_start=False, skip=1):
        obs = torch.tensor(obs, device='cpu')
        waypoints = torch.tensor(waypoints, device='cpu')

        if warm_start:
            self.actions.data[:-skip, :] = self.actions.clone().data[skip:, :]

            self.states.data[0] = obs
            for t in range(self.T):
                self.states.data[t + 1] = self.dynamics_model.discrete_dynamics(self.states[t], self.actions[t])

            self.lambdas = torch.tensor(data=torch.ones(self.T).float() * 5,
                                        device=torch.device('cpu'))
        else:
            self.actions = torch.tensor(data=torch.FloatTensor(self.action_dim, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cpu'), requires_grad=False)

        state_optimizer = torch.optim.Adam({self.states}, lr=self.lr)

        for i in range(self.iters):
            for j in range(500):
                state_optimizer.zero_grad()

                self.states.data[0] = obs

                next_states = self.dynamics_model.discrete_dynamics(self.states[self.initial_indices], self.actions)
                rewards = torch.tensor(0.0, device=torch.device('cpu')).repeat(self.T)
                rewards += (self.states[self.final_indices, 0] - waypoints[:self.T, 0]) ** 2 + (self.states[self.final_indices, 1] - waypoints[:self.T, 1]) ** 2
                rewards += self.lambdas * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) - self.epsilon)
                rewards += 0.5 * self.rho * (torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) ** 2)
                rewards = torch.sum(rewards)
                rewards.backward()
                torch.nn.utils.clip_grad_value_(self.states, 100.0)

                state_optimizer.step()
                self.get_actions_from_states(i)
                self.states.data[0] = obs

                with torch.no_grad():
                    self.states.data[:, 2] = torch.clamp(self.states[:, 2], 0.0, 20.0)
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)
                    self.actions.data[:, 1:] = torch.clamp(self.actions[:, 1:], 0.0, 1.0)
            print(self.states)
            self.states.data[0] = obs
            next_states = self.dynamics_model.discrete_dynamics(self.states[self.initial_indices], self.actions)
            self.lambdas.data += 0.01 * torch.log(torch.sum((self.states[self.final_indices] - next_states) ** 2, dim=1) / self.epsilon + 0.01) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000000.0)

        return self.states[:, :2].detach().numpy(), self.actions[0]

    def get_actions_from_states(self, i):
        self.actions = self.action_optimizer_model(self.states[self.initial_indices].cuda(),self.states[self.final_indices].cuda())
        self.actions = self.actions.cpu()

if __name__ == "__main__":
    env = CarEnv()
    state, waypoints = env.reset()

    model = DynamicsModel()

    done = False

    policy = LearntModelCollocationAmortizedPolicy(model)

    try:
        for i in range(500):
            if(i>0 and i<55):
                action = torch.tensor([0.0,1.0,0.0])
            else:
                states, action = policy(state, waypoints, warm_start=True)

            state, waypoints, done, _ = env.step(action, states)

    finally:
        print('destroying actors')
        for actor in env.actor_list:
            actor.destroy()