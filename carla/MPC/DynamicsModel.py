import torch
import pickle


class DynamicsModel:

    def __init__(self):
        self.DT = 0.1# [s] delta time step, = 1/FPS_in_server
        MODEL_NAME = "bicycle_model_100ms_20000_v4_jax"
        model_path="../SystemID/model/net_{}.model".format(MODEL_NAME)
        self.NN_W1, self.NN_W2, self.NN_W3, self.NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))
        self.NN_W1, self.NN_W2, self.NN_W3, self.NN_LR_MEAN = torch.tensor(self.NN_W1, device='cpu'), torch.tensor(self.NN_W2, device='cpu'), torch.tensor(self.NN_W3, device='cpu'), torch.tensor(self.NN_LR_MEAN, device='cpu')

    def NN3(self, x):
        x = torch.tanh(self.NN_W1 @ x.T)
        x = torch.tanh(self.NN_W2 @ x)
        x = self.NN_W3 @ x

        return x.T

    def continuous_dynamics(self, state, u):
        # state = [x, y, v, phi, beta, u]
        v = state[..., 2:3]
        v_sqrt = torch.sqrt(v)
        phi = state[..., 3:4]
        beta = state[..., 4:5]
        steering = torch.sin(u[..., 0:1])
        throttle_brake = torch.sin(u[..., 1:]) * 0.5 + 0.5

        deriv_x = v * torch.cos(phi + beta)
        deriv_y = v * torch.sin(phi + beta)
        deriv_phi = v * torch.sin(beta) / self.NN_LR_MEAN

        x1 = torch.hstack((
                    v_sqrt,
                    torch.cos(beta),
                    torch.sin(beta),
                    steering,
                    throttle_brake
                ))

        x2 = torch.hstack((
                    v_sqrt,
                    torch.cos(beta), 
                    -torch.sin(beta),
                    -steering,
                    throttle_brake
                ))

        x1 = self.NN3(x1)
        x2 = self.NN3(x2)

        deriv_v = (x1[..., 0:1] * (2 * v_sqrt.reshape(-1, 1) + x1[..., 0:1]) + x2[..., 0:1] * (2 * v_sqrt.reshape(-1, 1) + x2[..., 0:1])) / 2 # x1[0]+x2[0]
        deriv_beta = (x1[..., 1:2] - x2[..., 1:2]) / 2
        derivative = torch.hstack((deriv_x.reshape(-1, 1), deriv_y.reshape(-1, 1), deriv_v.reshape(-1, 1) / self.DT, deriv_phi.reshape(-1, 1), deriv_beta.reshape(-1, 1) / self.DT))

        return derivative.squeeze(0)

    def discrete_dynamics(self, state, u):
        return state + self.continuous_dynamics(state, u) * self.DT
