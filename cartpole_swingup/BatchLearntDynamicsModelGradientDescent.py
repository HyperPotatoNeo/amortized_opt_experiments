import gym
import modified_gym_cartpole_swingup
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)

from dynamics_model import MLP_Dynamics


class BatchLearntModelGradientDescentPolicy:

    def __init__(self, learnt_model, x_mean, x_std, y_mean, y_std, T=70, iters=20, lr=0.01, N=1):
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
        self.iters = iters
        self.lr = lr
        self.N = N
        self.actions = torch.tensor(data=torch.FloatTensor(self.N, self.T).uniform_(-1.0, 1.0), device=torch.device('cuda:0'), requires_grad=True)

    def __call__(self, state, warm_start=False):
        if warm_start:
            self.actions = torch.nn.Parameter(data=torch.hstack((self.actions[:, 1:], torch.zeros((self.N, 1), device=torch.device('cuda:0')))), requires_grad=True)
        else:
            self.actions = torch.tensor(data=torch.FloatTensor(self.N, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cuda:0'), requires_grad=True)

        optimizer = torch.optim.Adam({self.actions}, lr=self.lr)

        for i in range(self.iters):
            optimizer.zero_grad()
            current_state = state
            rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N)
            for t in range(self.T):
                current_state = current_state + (self.dynamics_model(((torch.hstack([current_state, self.actions[:, t].reshape(self.N, 1)]).float() - self.x_mean) / self.x_std).float()) * self.y_std) + self.y_mean
                reward = current_state[:, 2].cos() - abs(current_state[:, 0])
                rewards = rewards - reward
            rewards = torch.sum(rewards)
            rewards.backward()
            optimizer.step()

        return self.actions


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 20
    env.mpc_reset(N)

    data = np.load('dynamics_data/data_T_15_dagger.npy')
    train_x = data[:, :5]
    train_y = data[:, 5:]

    x_mean = torch.tensor(train_x.mean(axis=0), device=torch.device('cuda:0'))
    x_std = torch.tensor(train_x.std(axis=0), device=torch.device('cuda:0'))
    ltrain_y = train_y - train_x[:, :4]
    y_mean = torch.tensor(ltrain_y.mean(axis=0), device=torch.device('cuda:0'))
    y_std = torch.tensor(ltrain_y.std(axis=0), device=torch.device('cuda:0'))

    model = MLP_Dynamics().cuda()
    model.load_state_dict(torch.load('dynamics_models/dm_1.zip').state_dict())

    done = False

    policy = BatchLearntModelGradientDescentPolicy(model, x_mean, x_std, y_mean, y_std, N=N)

    for i in range(500):
        env.step(policy(env.state.detach(), warm_start=True))
        if i % 5 == 0:
            env.render()
