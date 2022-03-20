import torch
import numpy as np

class MLP_direct_policy_model(torch.nn.Module):
    def __init__(self, obs_dim=5, action_dim=1, T=40):
        super(MLP_direct_policy_model, self).__init__()

        self.linear1 = torch.nn.Linear(obs_dim, 512)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(512,256)
        self.linear3 = torch.nn.Linear(256, T)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = torch.tanh(x)
        return x

class MLP_direct_policy:
    def __init__(self, obs_dim=5, action_dim=1, T=40, model=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.T = T
        if(model==None):
            self.model = MLP_direct_policy_model(obs_dim, action_dim, T)
        else:
            self.model = model

    def __call__(self, obs):
        obs = torch.tensor(obs)
        obs = torch.unsqueeze(obs,0)
        actions = self.model(obs)
        return actions.numpy()