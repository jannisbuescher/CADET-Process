import matplotlib.pyplot as plt
import numpy as np

data = np.load('./CADETProcess/nn/data/data_100.npy')

plt.plot(data[98,12:])
plt.show()

plt.plot(data[98,2:])
plt.show()

for i in range(100):
    print(data[i,0:2])