# coding: utf-8
import time
import numpy as np
import gym
import torch
import modified_gym_cartpole_swingup
torch.set_printoptions(sci_mode=False)


class BatchCollocationSecondOrderPolicy:

    def __init__(self,  T=70, iters=10, lr=0.02, N=1):
        self.T = T
        self.iters = iters
        self.lr = lr
        self.N = N
        self.rho = 0
        self.damping = 1
        self.damped_eye = torch.eye((self.T)*4 + self.T).cuda()
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


    def LM_opt_step(self, J, residuals, states_actions):
        with torch.no_grad():
            JTJ = torch.t(J) @ J
            dx = torch.linalg.inv(JTJ + self.damping*torch.diag(torch.diag(JTJ))) @ torch.t(J) @ residuals
            states_actions[4:] = states_actions[4:] - self.lr*dx
            self.states.data = torch.reshape(states_actions[:(self.T + 1)*4], (self.T + 1, 4))
            self.actions.data = torch.squeeze(states_actions[(self.T + 1)*4:])


    def get_res(self, params):
        states = torch.reshape(params[:(self.T + 1)*4],(self.T + 1, 4))
        actions = params[(self.T + 1)*4:]
        self.mpc_env.mpc_reset(state=states[self.initial_indices])
        self.mpc_env.step(actions.reshape(self.N * self.T, 1))

        rew_t = states[self.final_indices, 2].cos() - abs(states[self.final_indices, 0])
        reward_res = torch.unsqueeze(torch.nn.functional.softplus(-torch.sum(rew_t)+70), dim=0)
        dyn_res = torch.flatten(torch.t(torch.sqrt(torch.tile(self.lambdas,(4,1))))*(states[self.final_indices] - self.mpc_env.state)) #+ np.sqrt(0.5 * self.rho)*(states[self.final_indices] - self.mpc_env.state))
        res = torch.cat([reward_res, dyn_res])
        #res = reward_res + torch.sum(dyn_res**2)
        return res


    def __call__(self, state, warm_start=False, skip=1):
        if warm_start:
            self.actions.data[:-skip] = self.actions.clone().data[skip:]

            self.mpc_env.mpc_reset(state=state)
            self.states.data[::self.T + 1] = self.mpc_env.state
            for t in range(self.T):
                self.mpc_env.step(self.actions[t::self.T].reshape(self.N, 1))
                self.states.data[t + 1::self.T + 1] = self.mpc_env.state

            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float() * 0.1,
                                        device=torch.device('cuda:0'))
        else:
            self.states.data *= 0.0
            self.states.data[::self.T + 1] = state
            self.actions.data = torch.tensor(data=torch.FloatTensor(self.N * self.T).uniform_(-1.0, 1.0),
                                             device=torch.device('cuda:0'), requires_grad=True)
            self.lambdas = torch.tensor(data=torch.ones(self.N * self.T).float(),
                                        device=torch.device('cuda:0'))


        for i in range(self.iters):
            
            for j in range(25):
                self.states.data[::self.T + 1] = state
                states_actions = torch.cat([torch.flatten(self.states),self.actions])
                J = torch.autograd.functional.jacobian(self.get_res, states_actions).detach()
                #print(J,J.shape)
                residuals = self.get_res(states_actions)
                residuals = torch.unsqueeze(residuals, dim=1)
                print('RESIDUALS = ',torch.sum(residuals**2))
                print('DYNAMICS RESIDUALS = ', torch.sum(residuals[1:]**2))
                print('REWARD RESIDUALS = ',residuals[0]**2)
                states_actions = torch.unsqueeze(states_actions, dim=1)
                self.LM_opt_step(J[:,4:], residuals, states_actions)

                res_cur = torch.sum(residuals**2).detach().cpu().numpy()
                if(j>0):
                    if(res_cur<res_prev):
                        self.damping = self.damping*0.8
                    else:
                        self.damping = self.damping*10
                self.damping = np.clip(self.damping, 0.0001, 5000)
                res_prev = res_cur
                #print('DAMPING',self.damping)
                rewards = torch.tensor(0.0, device=torch.device('cuda:0')).repeat(self.N * self.T)
                rewards += self.states[self.final_indices, 2].cos() - abs(self.states[self.final_indices, 0])
                if(j==99):
                    pass
                print('PRED REWARD = ',torch.sum(rewards))
                
                with torch.no_grad():
                    self.actions.data = torch.clamp(self.actions, -1.0, 1.0)
            print('DAMPING:',self.damping)
            self.states.data[::self.T + 1] = state
            self.mpc_env.mpc_reset(state=self.states[self.initial_indices])
            self.mpc_env.step(self.actions.reshape(self.N * self.T, 1))
            self.lambdas.data += 0.05 * torch.log(torch.sum((self.states[self.final_indices] - self.mpc_env.state) ** 2, dim=1) / (5e-6) + 0.01) * self.lambdas
            self.lambdas.data = torch.clamp(self.lambdas, 0.0, 500000000.0)
            print(self.states)
            print(self.actions)
            #COMPUTE TRUE REWARD
            states_copy = torch.clone(self.states).detach()
            self.mpc_env.mpc_reset(state=state)
            for t in range(self.T):
                self.mpc_env.step(self.actions[t::self.T].reshape(self.N, 1))
                self.states.data[t + 1::self.T + 1] = self.mpc_env.state
            rewards = self.states[self.final_indices, 2].cos() - abs(self.states[self.final_indices, 0])
            print('TRUE REWARD: ',torch.sum(rewards))
            self.states.data = states_copy
            print('STEP: ',i)
            #exit()

        return self.states, self.actions


if __name__ == "__main__":
    env = gym.make('ModifiedTorchCartPoleSwingUp-v0')
    N = 1
    T = 70
    env.reset()
    env.mpc_reset(N)

    done = False

    policy = BatchCollocationSecondOrderPolicy(T=T, iters=100, N=N)

    for i in range(500):
        states, actions = policy(env.state.detach(), warm_start=True, skip=10)
        for j in range(10):
            env.step(actions[j::T].reshape(N, 1))
            env.render()
