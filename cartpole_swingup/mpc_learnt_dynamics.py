# coding: utf-8
import gym
import numpy as np
import torch
from dynamics_model import MLP_Dynamics
import gym_cartpole_swingup

# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("CartPoleSwingUp-v0")
env.reset()

T = 60
actions = []
done = False

for i in range(T):
    actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))

train_x = np.load('data_T_15.npy')[:,:,:5]
train_x = train_x.reshape(len(train_x) * len(train_x[0]), 5)
train_y = np.load('data_T_15.npy')[:,:,5:]
train_y = train_y.reshape(len(train_y) * len(train_y[0]), 4)

for _ in range(50):
    device = torch.device('cpu')
    model = MLP_Dynamics()
    model.load_state_dict(torch.load('dynamics_models/dm_2.zip', map_location=device).state_dict())
    model.train(False)

    env.reset()
    done = False
    while not done:
        actions = []
        for i in range(T):
            actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
        # actions = actions[1:]
        # actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
        optimizer = torch.optim.Adam(actions, lr=0.01)

        for i in range(100):
            optimizer.zero_grad()
            current_state = torch.as_tensor(env.state)
            rewards = 0
            for t in range(T):
                current_state = model(torch.hstack([current_state, actions[t]]).float())
                reward = current_state[2].cos() - abs(current_state[0])
                rewards -= reward
            rewards.backward()
            optimizer.step()

        train_x = np.vstack((train_x, np.hstack((np.array(env.state), actions[0].detach().numpy()))))
        obs, rew, done, info = env.step(actions[0].detach().numpy())
        train_y = np.vstack((train_y, np.array(env.state)))
        print(model(torch.as_tensor(train_x[-1]).float()).detach().numpy() - train_y[-1])

        if np.random.rand() > 0.9:
            env.render()

    model = MLP_Dynamics().cuda()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(10000):
        model_optimizer.zero_grad()
        ltrain_x = np.array(train_x)
        ltrain_x = ltrain_x.reshape((len(train_x), 5))
        y_pred = model(torch.as_tensor(ltrain_x).cuda().float())
        ltrain_y = np.array(train_y)
        loss = loss_fn(y_pred, torch.as_tensor(ltrain_y).cuda().float())
        loss.backward()
        print(loss)
        model_optimizer.step()

    torch.save(model, 'dynamics_models/dm_2.zip')
