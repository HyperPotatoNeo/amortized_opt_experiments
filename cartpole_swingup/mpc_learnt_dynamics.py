# coding: utf-8
import gym
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
from dynamics_model import MLP_Dynamics
import gym_cartpole_swingup
import modified_gym_cartpole_swingup

def reset_environment(env, state):
    env.reset()
    env.state = state

# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
env = gym.make("CartPoleSwingUp-v0")
modified_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
env.reset()

T = 70
actions = []
done = False

for i in range(T):
    actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))

train_x = np.load('dynamics_data/data_T_15_dagger.npy')[:, :5]
train_y = np.load('dynamics_data/data_T_15_dagger.npy')[:, 5:]

for _ in range(5):
    train_x_mean = torch.tensor(train_x.mean(axis=0))
    train_x_std = torch.tensor(train_x.std(axis=0))
    ltrain_y = train_y - train_x[:, :4]
    train_y_mean = torch.tensor(ltrain_y.mean(axis=0))
    train_y_std = torch.Tensor(ltrain_y.std(axis=0))

    device = torch.device('cpu')
    model = MLP_Dynamics()
    model.load_state_dict(torch.load('dynamics_models/dm_1.zip', map_location=device).state_dict())
    model.train(False)

    env.reset()
    done = False
    for e in range(500):
        if done:
            break
        # actions = []
        # for i in range(T):
        #     actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
        actions = actions[1:]
        actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
        optimizer = torch.optim.Adam(actions, lr=0.01)

        for i in range(20):
        #     optimizer.zero_grad()
        #     reset_environment(modified_env, torch.as_tensor(env.state))
        #     rewards = 0
        #     for t in range(T):
        #         modified_env.state, reward, done, _ = modified_env.mpc_step(modified_env.state, actions[t])
        #         rewards -= reward
        #     rewards.backward(retain_graph=True)
        #     optimizer.step()
        #
        # print('best reward:', rewards)
        #
        # current_state = torch.as_tensor(env.state)
        # rewards = 0
        # for t in range(T):
        #     current_state = current_state + (model(((torch.hstack([current_state, actions[t]]) - train_x_mean) / train_x_std).float()) * train_y_std) + train_y_mean
        #     reward = current_state[2].cos() - abs(current_state[0])
        #     rewards -= reward
        # print('mid reward:', rewards)
        #
        # actions = []
        # for i in range(T):
        #     actions.append(torch.tensor(env.action_space.sample(), requires_grad=True))
        #
        # optimizer = torch.optim.Adam(actions, lr=0.01)
        #
        # for i in range(1000):
            optimizer.zero_grad()
            current_state = torch.as_tensor(env.state)
            rewards = 0
            for t in range(T):
                current_state = current_state + (model(((torch.hstack([current_state, actions[t]]).float() - train_x_mean) / train_x_std).float()) * train_y_std) + train_y_mean
                reward = current_state[2].cos() - abs(current_state[0])
                rewards -= reward
            rewards.backward()
            optimizer.step()

        train_x = np.vstack((train_x, np.hstack((np.array(env.state), actions[0].detach().numpy()))))
        obs, rew, done, info = env.step(actions[0].detach().numpy())
        train_y = np.vstack((train_y, np.array(env.state)))
        # print(torch.tensor(train_x[-1,:4]) + model(((torch.as_tensor(train_x[-1]) - train_x_mean) / train_x_std).float()).detach() * train_y_std + train_y_mean - torch.tensor(train_y[-1]))

        if e % 5 == 0:
            env.render()

    model = MLP_Dynamics().cuda()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    ltrain_y = train_y - train_x[:, :4]
    ltrain_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
    ltrain_y = (ltrain_y - ltrain_y.mean(axis=0)) / ltrain_y.std(axis=0)

    for epoch in range(30000):
        if epoch % 10000 == 0:
            model_optimizer.param_groups[0]['lr'] /= 10.0

        model_optimizer.zero_grad()
        y_pred = model(torch.as_tensor(ltrain_x).cuda().float())
        loss = loss_fn(y_pred, torch.as_tensor(ltrain_y).cuda().float())
        loss.backward()
        if epoch % 100 == 0:
            print(loss)
        model_optimizer.step()

    torch.save(model, 'dynamics_models/dm_1.zip')
    np.save('dynamics_data/data_T_15_dagger.npy', np.hstack((train_x, train_y)))
