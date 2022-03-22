import numpy as np
import torch
import modified_gym_cartpole_swingup
import gym_cartpole_swingup
import os

def collect_states(policy, env, state_dim=4, obs_dim=5, eps=30, eps_len=200, filename='data', dagger_iter=0):
    print('COLLECTING DAGGER STATES: ',dagger_iter)
    data = np.zeros((eps*eps_len,state_dim+obs_dim))
    rows = 0

    for i in range(eps):
        print('Episode: ',i)
        obs = env.reset()
        for j in range(eps_len):
            data[rows,:state_dim] = env.state
            data[rows,state_dim:] = obs
            rows += 1
            state = torch.unsqueeze(torch.tensor(env.state), dim=0).float().cuda()
            actions,_ = policy(state, batch_size=1)
            action = actions[0,0].detach().cpu().numpy()
            obs, rewards, done, info = env.step(action)
            if(done==True):
                break
            if(i%30==0):
                env.render()
    data = data[:rows,:]
    filename = filename+'_'+str(dagger_iter)

    np.save('dagger_data/'+filename+'.npy', data)
    return 'dagger_data/'+filename


def label_states(policy, data_file, output_file, batch_size, state_dim=4, obs_dim=5, action_dim=1, T=60, prev_file=None):
    print('LABELLING DAGGER STATES:')
    data = np.load(data_file+'.npy')
    data_labels = np.zeros((data.shape[0],data.shape[1]+(action_dim*T)))
    data_labels[:,:data.shape[1]] = data
    idx = 0

    for i in range(data.shape[0]//batch_size):
        print(i,'/',data.shape[0]//batch_size)
        states = data[i*batch_size:i*batch_size+batch_size, :state_dim]
        actions = policy(torch.tensor(states).cuda())
        data_labels[i*batch_size:i*batch_size+batch_size, data.shape[1]:] = actions.detach().cpu().numpy()
        idx = i*batch_size+batch_size

    final = idx
    if(idx<data.shape[0]):
        idx = idx-batch_size+data.shape[0]%batch_size
        states = data[idx:data.shape[0], :state_dim]
        policy.N = states.shape[0]
        actions = policy(torch.tensor(states).cuda())
        data_labels[final:, data.shape[1]:] = actions[final-idx:].detach().cpu().numpy()

    if(not prev_file==None):
        data_labels_prev = np.load(prev_file+'.npy', allow_pickle=True)
        data_labels = np.vstack((data_labels_prev,data_labels))
    
    np.save(output_file+'.npy', data_labels)