import numpy as np
import matplotlib.pyplot as plt

data = np.load('dynamics_data/data_T_15.npy')

# data = data[abs(data[:, 0]) < 2.5]
# data = data[abs(data[:, 1]) < 6.0]
# data = data[abs(data[:, 2]) < 7.0]
# data = data[abs(data[:, 3]) < 15.0]
data = data[abs(data[:, 4]) < 2.5]

# plt.plot(data[:,0])
# plt.show()

np.save('dynamics_data/data_T_15_cleaned.npy', data)