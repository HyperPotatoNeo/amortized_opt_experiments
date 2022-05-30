from car_env_for_MPC import CarEnv
from DynamicsModel import DynamicsModel
from LearntDynamicsModelGradientDescent import LearntModelGradientDescentPolicy
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)


if __name__ == "__main__":
    env = CarEnv()
    model = DynamicsModel()
    done = False
    policy = LearntModelGradientDescentPolicy(model)
    eps = 5
    ep_len = 250

    noise_std = [0.4,0.3,0.2,0.1,0.0]
    data = np.zeros((0,13))

    for i in range(len(noise_std)):
        for j in range(eps):
            state, waypoints = env.reset()
            states, action = policy(state, waypoints, warm_start=True)
            for k in range(ep_len):
                if(k<25 and i==0 and j==0):
                    action = torch.tensor([0.0,1.0,0.0])
                    state, waypoints, done, _ = env.step(action, states)
                else:
                    states, action = policy(state, waypoints, warm_start=True)
                    
                    epsilon = np.random.normal(size=3)
                    action = action.detach().numpy() + noise_std[i]*epsilon
                    action[0] = np.clip(action[0],-1,1)
                    action[1:] = np.clip(action[1:],0,1)
                    prev_state = state
                    print(action)
                    state, waypoints, done, _ = env.step(action, states)
                    if(done):
                        break
                    concat_states = np.concatenate([prev_state,state,action],axis=None)
                    data = np.vstack([data,concat_states])
                    #print(data)
            np.save('data/dynamics_data1.npy',data)