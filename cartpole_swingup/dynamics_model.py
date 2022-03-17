import torch
import numpy as np

class MLP_Dynamics(torch.nn.Module):

    def __init__(self, state_dim=5, action_dim=1):
        super(MLP_Dynamics, self).__init__()

        self.linear1 = torch.nn.Linear(state_dim+action_dim, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, state_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x