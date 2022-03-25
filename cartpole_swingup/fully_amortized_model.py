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


class LSTM_direct_policy_model(torch.nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden_size=128, num_layers=1):
        super(LSTM_direct_policy_model, self).__init__()
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size=state_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.linear1 = torch.nn.Linear(hidden_size,action_dim)
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(state_dim, hidden_size)
        self.relu = torch.nn.ReLU()
        self.activation = torch.nn.Tanh()
        self.hn = None
        self.cn = None
        self.hidden_size = hidden_size

    def forward(self, x, begin_sequence=False, batch_size=512):
        x = torch.unsqueeze(x, dim=0)

        if(begin_sequence):
            x, (self.hn, self.cn) = self.lstm(x,
                                              (torch.zeros((self.num_layers, batch_size, self.hidden_size)).float().cuda(),
                                               torch.zeros((self.num_layers, batch_size, self.hidden_size)).float().cuda()))
        else:
            x, (self.hn, self.cn) = self.lstm(x, (self.hn, self.cn))
        h_out = self.hn[-1]
        
        #x = self.linear2(x)
        #x = self.relu(x)
        #x = self.linear3(x)
        #x = self.relu(x)
        x = self.linear1(h_out)
        x = self.activation(x)
        return x


class LSTM_direct_policy:
    def __init__(self, env, state_dim=4, action_dim=1, T=40, hidden_size=128, num_layers=1, model=None, N=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.T = T
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        self.N = N
        self.env = env

        if(model==None):
            self.model = LSTM_direct_policy_model(state_dim, action_dim, hidden_size, num_layers).cuda()
        else:
            self.model = model.cuda()

    def __call__(self, state, batch_size=512):
        #state = torch.tensor(state).float().cuda()
        self.env.state = state
        actions = torch.zeros((self.N,self.T)).float().cuda()
        total_reward = 0
        for t in range(self.T):
            #state = self.env.state
            if(t==0):
                action_t = self.model(self.env.state, begin_sequence=True, batch_size=batch_size)
            else:
                #state = torch.zeros(state.shape).float().cuda()
                action_t = self.model(self.env.state)
            #self.env.mpc_step(state, action_t)
            self.env.state, reward, done, _ = self.env.mpc_step(self.env.state, action_t)
            total_reward = total_reward-reward
            actions[:,t] = actions[:,t]+torch.squeeze(action_t, dim=1)

        return actions, total_reward


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