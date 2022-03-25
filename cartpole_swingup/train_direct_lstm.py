import numpy as np
import torch
import modified_gym_cartpole_swingup
import fully_amortized_model
import gym_cartpole_swingup
import collect_states_labels
import BatchGradientDescentPolicy
import gym

env = gym.make("ModifiedTorchCartPoleSwingUp-v0")
mpc_env = gym.make("ModifiedTorchCartPoleSwingUp-v0")

T = 40
actions = []
done = False

n_dagger = 10
epochs = 1
batch_size = 1024

state_dim = 4
obs_dim = 5

filename = 'lstm_dagger'
output_file = 'dagger_data/lstm_dagger_9_labels'
COLLECT_DATA = True

batch_size = 512
GD_policy = BatchGradientDescentPolicy.BatchGradientDescentPolicy(T=T, iters=600, N=2048)


epochs = 151
prev_file = None

LSTM_policy = fully_amortized_model.LSTM_direct_policy(mpc_env, state_dim=4, action_dim=1, T=T, hidden_size=64, num_layers=1, model=None, N=batch_size)
optimizer = torch.optim.Adam(LSTM_policy.model.parameters(), lr=0.00005)

for i in range(n_dagger):
    criterion = torch.nn.MSELoss()
    states_file = filename
    states_labels_file = filename
    
    if(COLLECT_DATA):
        LSTM_policy.N = 20 + i * 3
        with torch.no_grad():
            data_file = collect_states_labels.collect_states(LSTM_policy, env, state_dim=4, obs_dim=5, eps=20+i*3,
                                                             eps_len=400+i*10, filename=states_file, dagger_iter=i)
        #data_file = 'dagger_data/lstm_dagger_0'
        output_file = data_file+'_'+'labels'
        collect_states_labels.label_states(GD_policy, data_file, output_file, batch_size=2048, state_dim=4, obs_dim=5, action_dim=1, T=T, prev_file=prev_file)
        prev_file = output_file

    LSTM_policy = fully_amortized_model.LSTM_direct_policy(mpc_env, state_dim=4, action_dim=1, T=T, hidden_size=64, num_layers=1, model=None, N=batch_size)
    optimizer = torch.optim.Adam(LSTM_policy.model.parameters(), lr=0.00005)
    
    data = np.load(output_file+'.npy')

    for j in range(epochs):
        print('EPOCH: ',j)
        np.random.shuffle(data)
        total_loss = 0
        for k in range(data.shape[0]//batch_size):
            optimizer.zero_grad()
            state = data[k*batch_size:k*batch_size+batch_size,:state_dim]
            state = torch.tensor(state).float().cuda()

            actions, rewards = LSTM_policy(state)
            #mean_reward = torch.sum(rewards/batch_size)
            #mean_reward.backward()
            action_labels = data[k*batch_size:k*batch_size+batch_size, state_dim+obs_dim:state_dim+obs_dim+T]
            action_labels = torch.tensor(action_labels).float().cuda()
            action_labels = torch.clamp(action_labels, -1.0, 1.0)
            loss = criterion(torch.flatten(actions), torch.flatten(action_labels))
            loss.backward()
            optimizer.step()
            total_loss = total_loss+loss.detach().cpu().numpy()
            #print(k,'/',data.shape[0]//batch_size,' COST = ',loss.detach().cpu().numpy())
        print('EPOCH LOSS = ',total_loss)
        
        if j % 50 == 0 and not j == 0:
            optimizer.param_groups[0]['lr'] /= 10.0

    torch.save(LSTM_policy.model, 'fully_amortized_models/LSTM_Dagger_' + str(i))