import random

import torch
import numpy as np

class MLP_Dynamics(torch.nn.Module):

    def __init__(self, state_dim=4, action_dim=1):
        super(MLP_Dynamics, self).__init__()

        self.linear1 = torch.nn.Linear(state_dim+action_dim, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, state_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def main():
    data = np.load('dynamics_data/data_T_15_cleaned.npy')
    data = data.reshape(len(data), len(data[0]))

    X = torch.as_tensor(data[:,:5]).cuda()
    Y = torch.as_tensor(data[:,5:]).cuda()
    Y = Y - X[:,:4]

    training = random.sample(range(len(X)), int(0.8 * len(X)))
    validation = list(set(range(len(X))) - set(training))

    train_x = (X[training] - X[training].mean(axis=0)) / X[training].std(axis=0)
    train_y = (Y[training] - Y[training].mean(axis=0)) / Y[training].std(axis=0)

    validation_x = (X[validation] - X[training].mean(axis=0)) / X[training].std(axis=0)
    validation_y = (Y[validation] - Y[training].mean(axis=0)) / Y[training].std(axis=0)

    model = MLP_Dynamics().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(30000):

        if epoch % 10000 == 0:
            print('------------------------------------')
            optimizer.param_groups[0]['lr'] /= 10.0

        optimizer.zero_grad()

        y_pred = model(train_x.float())
        loss = loss_fn(y_pred, train_y.float())
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print('Training loss: ', loss)
            # y_pred = model(validation_x.float())
            # loss = loss_fn(y_pred, validation_y.float())
            # print('Validation loss: ', loss)

    torch.save(model, 'dynamics_models/dm_1.zip')

if __name__ == '__main__':
    main()
