import random

import torch
import numpy as np
import argparse

import sys
sys.path.append('../')

from configs import observation_dim, action_dim


class ActionOptimizer(torch.nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_size=600, num_hidden=2):
        super(ActionOptimizer, self).__init__()
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.linear1 = torch.nn.Linear(2 * observation_dim, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear5 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear6 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear7 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear8 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.linear_out = torch.nn.Linear(self.hidden_size, action_dim)
        self.activation = torch.nn.ReLU()

        self.batchnorm1 = torch.nn.BatchNorm1d(self.hidden_size)
        self.batchnorm2 = torch.nn.BatchNorm1d(self.hidden_size)
        self.batchnorm3 = torch.nn.BatchNorm1d(self.hidden_size)
        self.batchnorm4 = torch.nn.BatchNorm1d(self.hidden_size)
        self.batchnorm5 = torch.nn.BatchNorm1d(self.hidden_size)
        self.batchnorm6 = torch.nn.BatchNorm1d(self.hidden_size)
        self.batchnorm7 = torch.nn.BatchNorm1d(self.hidden_size)
        self.batchnorm8 = torch.nn.BatchNorm1d(self.hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.batchnorm1(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.batchnorm2(x)

        x = self.linear3(x)
        x = self.activation(x)
        x = self.batchnorm3(x)

        x = self.linear4(x)
        x = self.activation(x)
        x = self.batchnorm4(x)

        x = self.linear5(x)
        x = self.activation(x)
        x = self.batchnorm5(x)

        x = self.linear6(x)
        x = self.activation(x)
        x = self.batchnorm6(x)

        x = self.linear7(x)
        x = self.activation(x)
        x = self.batchnorm7(x)

        x = self.linear8(x)
        x = self.activation(x)
        x = self.batchnorm8(x)

        x = self.linear_out(x)
        return x


class LearnActionOptimizer:

    def __init__(self, observation_dim, action_dim, filename, learning_rate, epochs, decay_factor, decay_after, hidden_size=600, num_hidden=2, model_path=None, model=None):
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
        self.batch_size = 100000
        if self.model is None:
            self.model = ActionOptimizer(observation_dim=self.observation_dim,
                                         action_dim=self.action_dim,
                                         hidden_size=self.hidden_size,
                                         num_hidden=self.num_hidden).cuda()
        self.data = np.load(self.filename)

    def append_data(self, data):
        self.data = np.vstack((self.data, data))

    def learn_action_optimizer(self, new_model=False):
        if new_model:
            self.model = ActionOptimizer(observation_dim=self.observation_dim,
                                         action_dim=self.action_dim,
                                         hidden_size=self.hidden_size,
                                         num_hidden=self.num_hidden).cuda()

        x = np.hstack((self.data[:, :self.observation_dim], self.data[:, self.observation_dim + 1:]))
        x[:, 4:] = x[:, 4:] - x[:, :4]
        y = self.data[:, self.observation_dim].reshape(-1, 1)
        training = random.sample(range(len(x)), int(0.7 * len(x)))
        validation = list(set(range(len(x))) - set(training))

        # train_x = (x[training] - x[training].mean(axis=0)) / x[training].std(axis=0)
        # train_y = (y[training] - y[training].mean(axis=0)) / y[training].std(axis=0)
        #
        # validation_x = (x[validation] - x[training].mean(axis=0)) / x[training].std(axis=0)
        # validation_y = (y[validation] - y[training].mean(axis=0)) / y[training].std(axis=0)
        #
        # train = np.hstack((train_x, train_y))
        # validation = np.hstack((validation_x, validation_y))
        train = np.hstack((x[training], y[training]))
        validation = np.hstack((x[validation], y[validation]))

        del x, y

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        for epoch in range(self.epochs):
            if epoch % self.decay_after == 0:
                print('------------------------------------')
                optimizer.param_groups[0]['lr'] /= self.decay_factor

            np.random.shuffle(train)

            total_loss = 0
            for batch in range(int(len(train) / self.batch_size) + (len(train) / self.batch_size) > 0):
                optimizer.zero_grad()

                train_x = torch.tensor(train[batch * self.batch_size: (batch + 1) * self.batch_size, :self.observation_dim * 2], device='cuda:0').float()
                train_y = torch.tensor(train[batch * self.batch_size: (batch + 1) * self.batch_size, self.observation_dim * 2:], device='cuda:0').float()

                y_pred = self.model(train_x)
                loss = loss_fn(y_pred, train_y)
                total_loss += loss
                loss.backward()
                optimizer.step()

            if epoch % int(10) == 0:
                print(epoch)
                print('Training loss: ', total_loss)
                with torch.no_grad():
                    validation_sampled = validation[random.sample(range(len(validation)), int(0.05 * len(validation)))]
                    validation_x = torch.tensor(validation_sampled[:, :self.observation_dim * 2], device='cuda:0').float()
                    validation_y = torch.tensor(validation_sampled[:, self.observation_dim * 2:], device='cuda:0').float()
                    y_pred = self.model(validation_x)
                    loss = loss_fn(y_pred, validation_y)
                    print('Validation loss: ', loss)

                if epoch % 100 == 0:
                    if self.model_path is not None:
                        torch.save(self.model, self.model_path)

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

    learn_action_optimizer = LearnActionOptimizer(observation_dim=observation_dim(args.env_name),
                                                  action_dim=action_dim(args.env_name),
                                                  hidden_size=args.hidden_size,
                                                  num_hidden=args.num_hidden,
                                                  filename=args.filename,
                                                  learning_rate=args.learning_rate,
                                                  epochs=args.epochs,
                                                  decay_factor=args.decay_factor,
                                                  decay_after=args.decay_after,
                                                  model_path=args.model_path)
    learn_action_optimizer.learn_action_optimizer()


if __name__ == '__main__':
    main()
