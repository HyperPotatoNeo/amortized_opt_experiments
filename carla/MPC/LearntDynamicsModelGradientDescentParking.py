from car_env_for_MPC import CarEnv
from DynamicsModel import DynamicsModel
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)


class LearntModelGradientDescentPolicy:

    def __init__(self, learnt_model, T=45, iters=30, lr=0.005):
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
            self.T -= 1
            self.T = max(self.T, 5)
            previous = self.actions.clone().data[:, skip:]
            self.actions = torch.tensor(data=torch.FloatTensor(self.action_dim, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cpu'), requires_grad=True)
            self.actions.data[:, :len(previous[0])] = previous
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

                reward = 0
                if t > self.T - 8:
                    reward += 500 * (current_state[3] - 3.14) ** 2
                if t == self.T - 1:
                    reward += 50 * ((current_state[0] - waypoints[0]) ** 2 + 2 * (current_state[1] - waypoints[1]) ** 2)
                    reward += 50 * (current_state[2]) ** 2

                if i == self.iters - 1:
                    states[t + 1] = current_state[:2].cpu().detach().numpy()

                reward -= 16 * (torch.clip((current_state[0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -41.16636276245117) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -38.57696533203125) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -35.526119232177734) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -32.72093963623047) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -30.194425582885742) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -21.28460693359375) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -28.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -18.913570404052734) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -32.89897537231445) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -35.526119232177734) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -30.194425582885742) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -12.52372932434082) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -18.60922622680664) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -21.28460693359375) ** 2 / 16, 0, 1))
                reward -= 16 * (torch.clip((current_state[0] - -22.6) ** 2 / 25, 0, 1) + torch.clip((current_state[1] - -15.445639610290527) ** 2 / 16, 0, 1))

                rewards = rewards + reward

            rewards = torch.sum(rewards)
            rewards.backward()
            optimizer.step()
            with torch.no_grad():
                self.actions.data = torch.clamp(self.actions, -1.0, 1.0)
                self.actions.data[1:, :] = torch.clamp(self.actions[1:, :], 0.0, 1.0)

        return states, states, self.actions[:, 0]


if __name__ == "__main__":
    env = CarEnv()
    state, waypoints = env.reset(parking=True)

    model = DynamicsModel()

    done = False

    policy = LearntModelGradientDescentPolicy(model)

    target = [-23.465097427368164, -18.80262565612793]

    try:
        for i in range(500):
            states, states, action = policy(state, target, warm_start=True)

            state, waypoints, done, _ = env.step(action, states, actual_states=states)
            print(state)

    finally:
        print('destroying actors')
        for actor in env.actor_list:
            actor.destroy()