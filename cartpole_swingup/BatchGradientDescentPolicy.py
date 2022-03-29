# coding: utf-8
import gym
import torch
import modified_gym_cartpole_swingup
torch.set_printoptions(sci_mode=False)


class BatchGradientDescentPolicy:

    def __init__(self,  T=70, iters=10, lr=0.01, N=1):
        self.T = T
        self.iters = iters
        self.lr = lr
        self.N = N
        self.mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
        self.mpc_env.mpc_reset(self.N)
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
            self.mpc_env.mpc_reset(state=state)
            rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N)
            for t in range(self.T):
                obs, reward, done, _ = self.mpc_env.step(self.actions[:, t].reshape(self.N, 1))
                rewards -= reward
            rewards = torch.sum(rewards)
            rewards.backward()
            optimizer.step()
            with torch.no_grad():
                self.actions.data = torch.clamp(self.actions, -1.0, 1.0)

        return self.actions


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 20
    env.mpc_reset(N)

    done = False

    policy = BatchGradientDescentPolicy(N=N)

    for i in range(500):
        env.step(policy(env.state.detach(), warm_start=True))
        if i % 5 == 0:
            env.render()
