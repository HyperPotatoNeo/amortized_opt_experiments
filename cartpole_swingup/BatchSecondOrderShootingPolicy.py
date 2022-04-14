# coding: utf-8
import gym
import torch
import modified_gym_cartpole_swingup
torch.set_printoptions(sci_mode=False)


class BatchSecondOrderShootingPolicy:

    def __init__(self,  T=70, iters=10, lr=0.01, N=1):
        self.T = T
        self.iters = iters
        self.lr = lr
        self.N = N
        self.mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
        self.mpc_env.mpc_reset(self.N)
        self.actions = torch.tensor(data=torch.FloatTensor(self.N, self.T).uniform_(-1.0, 1.0), device=torch.device('cuda:0'), requires_grad=True)
        self.damped_eye = 0.01 * torch.eye(self.T).cuda()


    def LM_opt_step(self, J, residuals, actions):
        dx = torch.linalg.inv(torch.t(J) @ J + self.damped_eye) @ torch.t(J) @ residuals
        print(residuals)
        actions = actions - self.lr*dx
        self.actions.data = torch.reshape(actions, (self.N, self.T))


    def get_res(self, actions):
        reward_res = torch.zeros(1).cuda()
        for t in range(self.T):
            obs, reward, done, _ = self.mpc_env.step(actions[t].reshape(self.N, 1))
            reward_res += reward[0]
        reward_res = torch.nn.functional.softplus(-reward_res+70)
        
        return reward_res


    def __call__(self, state, warm_start=False):
        if warm_start:
            self.actions = torch.nn.Parameter(data=torch.hstack((self.actions[:, 1:], torch.zeros((self.N, 1), device=torch.device('cuda:0')))), requires_grad=True)
        else:
            self.actions = torch.tensor(data=torch.FloatTensor(self.N, self.T).uniform_(-1.0, 1.0),
                                        device=torch.device('cuda:0'), requires_grad=True)


        for i in range(self.iters):
            self.mpc_env.mpc_reset(state=state)
            actions = torch.flatten(self.actions)
            #print('HERE')
            J = torch.autograd.functional.jacobian(self.get_res, actions)
            self.mpc_env.mpc_reset(state=state)
            residuals = self.get_res(actions)
            residuals = torch.unsqueeze(residuals, dim=1)
            actions = torch.unsqueeze(actions, dim=1)
            self.LM_opt_step(J, residuals, actions)

            rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N)
            self.mpc_env.mpc_reset(state=state)
            for t in range(self.T):
                obs, reward, done, _ = self.mpc_env.step(self.actions[:, t].reshape(self.N, 1))
                rewards += reward
            print('REWARD = ',rewards)

            with torch.no_grad():
                self.actions.data = torch.clamp(self.actions, -1.0, 1.0)

        return self.actions


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 1
    env.mpc_reset(N)

    done = False

    policy = BatchSecondOrderShootingPolicy(N=N, iters=5)

    for i in range(500):
        print('STEP: ',i)
        actions = policy(env.state.detach(), warm_start=True)
        print(actions)
        for j in range(1):
            env.step(actions[:,j].reshape(1, 1))
        env.render()
