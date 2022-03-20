import numpy as np
import torch
import gym
import modified_gym_cartpole_swingup
import fully_amortized_model
import gym_cartpole_swingup
import os

def reset_environment(env, state):
    env.reset()
    env.state = state


def collect_data(policy, env, state_dim=4, obs_dim=5, eps=150, eps_len=200, filename='data', dagger_iter=0):
    print('DAGGER ITER: ',dagger_iter)
    data = np.zeros((eps*eps_len,state_dim+obs_dim))
    rows = 0

    for i in range(eps):
        if(i%20==0):
            print('Episode: ',i)
        obs = env.reset()
        for j in range(eps_len):
            data[rows,:state_dim] = env.state
            data[rows,state_dim:] = obs
            rows += 1
            actions = policy(obs)
            action = actions[0,0]
            obs, rewards, done, info = env.step(action)
            if(done==True):
                break
            if(i%50==0):
                env.render()
    data = data[:rows,:]

    prev_filename = filename+'_'+str(dagger_iter-1)+'.npy'
    filename = filename+'_'+str(dagger_iter)

    if(os.path.exists('dagger_data/'+prev_filename)):
        data_prev = np.load('dagger_data/'+prev_filename, allow_pickle=True)
        data = np.vstack((data_prev,data))

    np.save('dagger_data/'+filename+'.npy', data)


env = gym.make("TorchCartPoleSwingUp-v0")
env.reset()
mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")

T = 40
actions = []
done = False

n_dagger = 5
epochs = 1
batch_size = 1024

state_dim = 4
obs_dim = 5

filename = 'data'

for i in range(n_dagger):
    policy = fully_amortized_model.MLP_direct_policy(obs_dim=obs_dim, action_dim=1, T=T, model=None)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=0.0005)

    with torch.no_grad():
        policy.model.eval()
        collect_data(policy, env, state_dim=4, obs_dim=obs_dim, eps=150, eps_len=200, filename=filename, dagger_iter=i)
    cur_file = filename+'_'+str(i)+'.npy'
    data = np.load('dagger_data/'+cur_file)
    policy.model.train()

    print('TRAINING MODEL:')

    NO_LABELS = True

    if(NO_LABELS):
        for j in range(epochs):
            print('EPOCH: ',j)
            np.random.shuffle(data)

            for k in range(data.shape[0]//batch_size):
                optimizer.zero_grad()

                obs = data[k*batch_size:k*batch_size+batch_size,state_dim:]
                obs = torch.tensor(obs).float()
                actions = policy.model(obs)
                print(actions[0].detach().numpy())
                total_cost = 0.0
                #if(k%50==0 and not k==0):
                #    optimizer.param_groups[0]['lr'] /= 5.0

                for ep in range(batch_size):
                    cur_state = data[k*batch_size+ep,:state_dim]
                    reset_environment(mpc_env, torch.as_tensor(env.state))
                    ep_cost = 0.0
                    for t in range(T):
                        mpc_env.state, reward, done, _ = mpc_env.mpc_step(mpc_env.state, torch.unsqueeze(actions[ep,t],0))
                        ep_cost = ep_cost - reward
                    total_cost = total_cost + ep_cost
                
                total_cost = total_cost/batch_size
                total_cost.backward(total_cost)
                optimizer.step()
                print('Batch: ',k,total_cost.detach().numpy())
        torch.save(policy.model, 'fully_amortized_models/MLP_Dagger_'+str(i))

    else:
        ###ORACLE LABELS MEAN SQUARED ERROR TRAINING####
        pass