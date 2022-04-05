import torch
import numpy as np
import pandas as pd

class MLP_direct_policy_model(torch.nn.Module):
    def __init__(self, state_dim=4):
        super(MLP_direct_policy_model, self).__init__()

        self.linear1 = torch.nn.Linear(state_dim*2, 256)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256,1)
        self.linear3 = torch.nn.Linear(256, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        #x = self.activation(x)
        #x = self.linear3(x)
        #x = self.sigmoid(x)
        return x


def train_feasibility_model(epochs, feasible_file='feasible_dynamics.csv', infeasible_file='infeasible_dynamics.csv', lr=0.0001, batch_size=2048, model_name='feasibility_model'):
    feasible = pd.read_csv('feasibility_data/'+feasible_file, header=None).values
    infeasible = pd.read_csv('feasibility_data/'+infeasible_file, header=None).values

    data = np.vstack([feasible,infeasible])
    data_mean = np.mean(data[:,:8], axis=0)
    data_std = np.std(data[:,:8], axis=0)

    data[:,:8] = (data[:,:8]-data_mean)/data_std
    data_statistics = pd.DataFrame(np.vstack([data_mean, data_std]))
    data_statistics.to_csv('feasibility_data/statistics.csv', header=False, index=False)

    model = MLP_direct_policy_model().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    epsilon = 0

    for i in range(epochs):
        print('EPOCH: ',i)
        np.random.shuffle(data)
        total_loss = 0
        for j in range(data.shape[0]//batch_size):
            optim.zero_grad()
            states = data[j*batch_size:j*batch_size+batch_size, :8]
            labels = data[j*batch_size:j*batch_size+batch_size, 8:]
            states = torch.tensor(states).float().cuda()
            labels = torch.tensor(labels).float().cuda()

            pred = model(states)
            loss = criterion(pred, labels)
            loss.backward()
            optim.step()
            total_loss += loss.detach().cpu().numpy()
        print('LOSS = ',total_loss)

        if(i%10==0):
            np.random.shuffle(data)
            acc = 0
            with torch.no_grad():
                for j in range(data.shape[0]//batch_size):
                    states = data[i*batch_size:i*batch_size+batch_size, :8]
                    labels = data[i*batch_size:i*batch_size+batch_size, 8:]
                    states = torch.tensor(states).float().cuda()
                    labels = torch.tensor(labels).float().cuda()
                    pred = torch.round(torch.sigmoid(model(states)))
                    acc += torch.sum(torch.abs(pred-labels)).detach().cpu().numpy()
            
                print('ACCURACY = ',1-acc/(batch_size*(data.shape[0]//batch_size)))

    torch.save(model, 'feasibility_data/feasibility_model')
    return model             


if __name__=='__main__':
    train_feasibility_model(epochs=100)