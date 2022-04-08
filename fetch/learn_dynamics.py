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


class LearnDynamics:

    def __init__(self, observation_dim, action_dim, filename, learning_rate, epochs, decay_factor, decay_after, hidden_size=200, num_hidden=1, model_path=None, model=None):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.filename = filename
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.decay_factor = decay_factor
        self.decay_after = decay_after
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.model_path = model_path
        self.model = model
        if self.model is None:
            self.model = MLP_Dynamics(observation_dim=observation_dim,
                                      action_dim=action_dim,
                                      hidden_size=hidden_size,
                                      num_hidden=num_hidden).cuda()
        self.data = np.load(self.filename)

    def append_data(self, data):
        self.data = np.vstack((self.data, data))

    def learn_dynamics(self, new_model=False):
        if new_model:
            self.model = MLP_Dynamics(observation_dim=self.observation_dim,
                                      action_dim=self.action_dim,
                                      hidden_size=self.hidden_size,
                                      num_hidden=self.num_hidden).cuda()

        X = torch.as_tensor(self.data[:, :self.observation_dim + self.action_dim]).float().cuda()
        Y = torch.as_tensor(self.data[:, self.observation_dim + self.action_dim:]).float().cuda()
        Y = Y - X[:, :self.observation_dim]

        training = random.sample(range(len(X)), int(0.8 * len(X)))
        validation = list(set(range(len(X))) - set(training))

        train_x = (X[training] - X[training].mean(axis=0)) / X[training].std(axis=0)
        train_y = (Y[training] - Y[training].mean(axis=0)) / Y[training].std(axis=0)

        validation_x = (X[validation] - X[training].mean(axis=0)) / X[training].std(axis=0)
        validation_y = (Y[validation] - Y[training].mean(axis=0)) / Y[training].std(axis=0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        del X, Y

        for epoch in range(self.epochs):
            if epoch % self.decay_after == 0:
                # print('------------------------------------')
                optimizer.param_groups[0]['lr'] /= self.decay_factor

            optimizer.zero_grad()

            y_pred = self.model(train_x)
            loss = loss_fn(y_pred, train_y)
            loss.backward()
            optimizer.step()
            # if epoch % int(self.epochs / 20) == 0:
            #     print(epoch)
            #     print('Training loss: ', loss)
            #     y_pred = self.model(validation_x)
            #     loss = loss_fn(y_pred, validation_y)
            #     print('Validation loss: ', loss)

        if self.model_path is not None:
            torch.save(self.model, self.model_path)
        return self.model


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

    learn_dyanamics = LearnDynamics(observation_dim=observation_dim(args.env_name),
                                    action_dim=action_dim(args.env_name),
                                    hidden_size=args.hidden_size,
                                    num_hidden=args.num_hidden,
                                    filename=args.filename,
                                    learning_rate=args.learning_rate,
                                    epochs=args.epochs,
                                    decay_factor=args.decay_factor,
                                    decay_after=args.decay_after,
                                    model_path=args.model_path)
    learn_dyanamics.learn_dynamics()


if __name__ == '__main__':
    main()
