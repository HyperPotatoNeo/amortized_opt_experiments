import torch
import numpy as np

class MLP_Dynamics(torch.nn.Module):

    def __init__(self, state_dim=4, action_dim=1):
        super(MLP_Dynamics, self).__init__()

        self.linear1 = torch.nn.Linear(state_dim+action_dim, 100)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, state_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def main():
    data = np.load('data_T_15.npy')
    data = data.reshape(len(data) * len(data[0]), len(data[0,0]))
    X = torch.as_tensor(data[:,:5]).cuda()
    Y = torch.as_tensor(data[:,5:]).cuda()

    model = MLP_Dynamics().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(10000):

        optimizer.zero_grad()

        y_pred = model(X.float())
        loss = loss_fn(y_pred, Y.float())
        loss.backward()
        optimizer.step()
        print(loss)

    torch.save(model, 'dynamics_models/dm_1.zip')

if __name__ == '__main__':
    main()
