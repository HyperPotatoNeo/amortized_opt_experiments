import random

import torch
import numpy as np
import argparse

import sys
sys.path.append('../')

from configs import observation_dim, action_dim


class MLP_Dynamics(torch.nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_size=200, num_hidden=1):
        super(MLP_Dynamics, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.linear1 = torch.nn.Linear(observation_dim + action_dim, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = torch.nn.Linear(self.hidden_size, observation_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        for _ in range(self.num_hidden - 1):
            x = self.linear2(x)
            x = self.activation(x)
        x = self.linear3(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='ModifiedTorchCartPoleSwingUp-v0', help='the environment name')
    parser.add_argument('--model-path', type=str, default='dynamics_models/dm_1.zip', help='Dynamics model saving path')
    parser.add_argument('--filename', type=str, default='dynamics_data/data_T_15.npy', help='File name having data')
    parser.add_argument('--epochs', type=int, default=30000, help='Number of epochs for training')
    parser.add_argument('--decay-after', type=int, default=10000, help='Decay learning rate after')
    parser.add_argument('--decay-factor', type=float, default=10.0, help='Learning rate decay factor')
    parser.add_argument('--num-hidden', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--hidden-size', type=int, default=200, help='Size of each hidden layer')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
    args = parser.parse_args()

    model = MLP_Dynamics(observation_dim=observation_dim(args.env_name),
                         action_dim=action_dim(args.env_name),
                         hidden_size=args.hidden_size,
                         num_hidden=args.num_hidden).cuda()

    data = np.load(args.filename)
    print(data.shape)
    data = data.reshape(len(data), len(data[0]))

    X = torch.as_tensor(data[:, :observation_dim(args.env_name) + action_dim(args.env_name)]).float().cuda()
    Y = torch.as_tensor(data[:, observation_dim(args.env_name) + action_dim(args.env_name):]).float().cuda()
    Y = Y - X[:, :observation_dim(args.env_name)]

    training = random.sample(range(len(X)), int(0.8 * len(X)))
    validation = list(set(range(len(X))) - set(training))

    train_x = (X[training] - X[training].mean(axis=0)) / X[training].std(axis=0)
    train_y = (Y[training] - Y[training].mean(axis=0)) / Y[training].std(axis=0)
    print(train_x.shape, Y[training].std(axis=0))

    validation_x = (X[validation] - X[training].mean(axis=0)) / X[training].std(axis=0)
    validation_y = (Y[validation] - Y[training].mean(axis=0)) / Y[training].std(axis=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()

    del X, Y

    for epoch in range(args.epochs):
        if epoch % args.decay_after == 0:
            print('------------------------------------')
            optimizer.param_groups[0]['lr'] /= args.decay_factor

        optimizer.zero_grad()

        y_pred = model(train_x)
        loss = loss_fn(y_pred, train_y)
        loss.backward()
        optimizer.step()
        print(epoch)
        print('Training loss: ', loss)
        # if epoch % int(args.epochs / 20) == 0:
        y_pred = model(validation_x)
        loss = loss_fn(y_pred, validation_y)
        print('Validation loss: ', loss)

    torch.save(model, args.model_path)


if __name__ == '__main__':
    main()
