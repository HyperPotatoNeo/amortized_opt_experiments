import gym
import gym_cartpole_swingup
import numpy as np
import torch

from dynamics_model import MLP_Dynamics


class LearntModelGradientDescentPolicy:

    def __init__(self, learnt_model, x_mean, x_std, y_mean, y_std, env, iters=20, lr=0.01, T=70):
        if isinstance(learnt_model, str):
            self.dynamics_model = MLP_Dynamics()
            self.dynamics_model.load_state_dict(torch.load(learnt_model).state_dict())
        elif isinstance(learnt_model, MLP_Dynamics):
            self.dynamics_model = learnt_model
        else:
            raise NotImplementedError
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.T = T
        self.env = env
        self.actions = []
        for _ in range(self.T):
            self.actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
        self.lr = lr
        self.iters = iters

    def __call__(self, state=None):
        if state is None:
            state = self.env.state

        self.actions = self.actions[1:]
        self.actions.append(torch.tensor(self.env.action_space.sample(), requires_grad=True))

        self.optimizer = torch.optim.Adam(self.actions, lr=self.lr)

        for i in range(self.iters):
            self.optimizer.zero_grad()
            current_state = torch.as_tensor(state)
            rewards = 0
            for t in range(self.T):
                current_state = current_state + (self.dynamics_model(((torch.hstack([current_state, self.actions[t]]).float() - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
                reward = current_state[2].cos() - abs(current_state[0])
                rewards -= reward
            rewards.backward()
            self.optimizer.step()

        return self.actions[0].detach().numpy()


if __name__ == "__main__":
    env = gym.make("CartPoleSwingUp-v0")
    env.reset()

    data = np.load('dynamics_data/data_T_15_dagger.npy')
    train_x = data[:, :5]
    train_y = data[:, 5:]

    x_mean = torch.tensor(train_x.mean(axis=0))
    x_std = torch.tensor(train_x.std(axis=0))
    ltrain_y = train_y - train_x[:, :4]
    y_mean = torch.tensor(ltrain_y.mean(axis=0))
    y_std = torch.Tensor(ltrain_y.std(axis=0))

    model = MLP_Dynamics()
    model.load_state_dict(torch.load('dynamics_models/dm_1.zip').state_dict())

    done = False

    policy = LearntModelGradientDescentPolicy(model, x_mean, x_std, y_mean, y_std, env)

    while not done:
        obs, rewards, done, info = env.step(policy())
        env.render()