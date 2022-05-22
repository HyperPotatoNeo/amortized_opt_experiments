from car_env_for_MPC import CarEnv
from DynamicsModel import DynamicsModel
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)


class LearntModelGradientDescentPolicy:

    def __init__(self, learnt_model, T=10, iters=30, lr=0.005):
        self.dynamics_model = learnt_model
        self.T = T
        self.iters = iters
        self.lr = lr
        self.action_dim = 3
        self.actions = torch.tensor(data=torch.FloatTensor(self.action_dim, self.T).uniform_(-1.0, 1.0),
                                    device=torch.device('cpu'), requires_grad=True)

    def __call__(self, obs, waypoints, warm_start=False, skip=1):
        obs = torch.tensor(obs, device='cpu')

        if warm_start:
            self.actions.data[:, :-skip] = self.actions.clone().data[:, skip:]
        else:
            self.actions = torch.tensor(data=torch.FloatTensor(self.action_dim, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cpu'), requires_grad=True)

        optimizer = torch.optim.Adam({self.actions}, lr=self.lr)


        for i in range(self.iters):
            optimizer.zero_grad()
            current_state = obs

            if i == self.iters - 1:
                states = np.zeros((self.T + 1, 2))
                states[0] = current_state[:2].cpu().detach().numpy()

            rewards = torch.tensor(0.0, device=torch.device('cpu'))
            for t in range(self.T):
                current_state = self.dynamics_model.discrete_dynamics(current_state.float(), self.actions[:, t].reshape(self.action_dim).float()).float()
                reward = (current_state[0] - waypoints[t + 1, 0]) ** 2 + (current_state[1] - waypoints[t + 1, 1]) ** 2
                rewards = rewards + reward

                if i == self.iters - 1:
                    states[t + 1] = current_state[:2].cpu().detach().numpy()

            rewards = torch.sum(rewards)
            rewards.backward()
            optimizer.step()
            with torch.no_grad():
                self.actions.data = torch.clamp(self.actions, -1.0, 1.0)
                self.actions.data[1:, :] = torch.clamp(self.actions[1:, :], 0.0, 1.0)

        return states, self.actions[:, 0]


if __name__ == "__main__":
    env = CarEnv()
    state, waypoints = env.reset()

    model = DynamicsModel()

    done = False

    policy = LearntModelGradientDescentPolicy(model)

    try:
        for i in range(500):
            states, action = policy(state, waypoints, warm_start=True)

            state, waypoints, done, _ = env.step(action, states)
            print(state, waypoints[:15])

    finally:
        print('destroying actors')
        for actor in env.actor_list:
            actor.destroy()