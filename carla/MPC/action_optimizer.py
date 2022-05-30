import random

import torch
import numpy as np
import argparse

import sys
sys.path.append('../')

class ActionOptimizer(torch.nn.Module):

    def __init__(self, observation_dim=5, action_dim=3, hidden_size=512, num_hidden=2):
        super(ActionOptimizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.linear1 = torch.nn.Linear(observation_dim, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.linear_out = torch.nn.Linear(self.hidden_size, action_dim)
        self.activation = torch.nn.ReLU()

        #self.NN_W1, self.NN_W2, self.NN_W3, self.NN_LR_MEAN = pickle.load(open(model_path, mode="rb"))
        #self.NN_W1, self.NN_W2, self.NN_W3, self.NN_LR_MEAN = torch.tensor(self.NN_W1, device='cuda'), torch.tensor(self.NN_W2, device='cuda'), torch.tensor(self.NN_W3, device='cuda'), torch.tensor(self.NN_LR_MEAN, device='cuda')


    def forward(self, state1, state2):
        x = (state2-state1)/0.1
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear_out(x)
        return x


def train_IDM(epochs, lr=0.001, batch_size=128, filename='semi_feasible_dynamics.csv'):
    train_val_split = 0.9
    data = np.load('data/dynamics_data.npy')

    model = ActionOptimizer().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 2000, 0.1)
    criterion = torch.nn.MSELoss()
    np.random.shuffle(data)
    train_data = data[:int(data.shape[0]*train_val_split)]
    val_data = data[int(data.shape[0]*train_val_split):]
    best_val = 100

    for i in range(epochs):
        print('EPOCH: ',i)
        np.random.shuffle(train_data)
        total_loss = 0
        model.train()
        for j in range(train_data.shape[0]//batch_size):
            optim.zero_grad()
            states = train_data[j*batch_size:j*batch_size+batch_size, :10]
            labels = train_data[j*batch_size:j*batch_size+batch_size, 10:]
            states = torch.tensor(states).float().cuda()
            labels = torch.tensor(labels).float().cuda()

            pred = model(states[:,:5],states[:,5:10])
            loss = criterion(pred, labels)
            loss.backward()
            optim.step()
            total_loss += loss.detach().cpu().numpy()
        print('TRAIN LOSS = ',total_loss/(train_data.shape[0]//batch_size))

        total_loss = 0
        model.eval()
        with torch.no_grad():
            for j in range(val_data.shape[0]//batch_size):
                states = val_data[j*batch_size:j*batch_size+batch_size, :10]
                labels = val_data[j*batch_size:j*batch_size+batch_size, 10:]
                states = torch.tensor(states).float().cuda()
                labels = torch.tensor(labels).float().cuda()
                pred = model(states[:,:5],states[:,5:10])
                loss = criterion(pred, labels)
                total_loss += loss.detach().cpu().numpy()
        print('VALIDATION LOSS = ',total_loss/(val_data.shape[0]//batch_size))
        lr_scheduler.step()
        if(total_loss/(val_data.shape[0]//batch_size)<best_val):
            best_val = total_loss/(val_data.shape[0]//batch_size)
            torch.save(model, '../SystemID/inverse_model/action_model')
    return model 

if __name__=='__main__':
    train_IDM(10000)            