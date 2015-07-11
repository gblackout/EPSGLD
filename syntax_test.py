import numpy as np
import h5py



def p(data, li, axis=1):
    for start, end in li:
        if axis: yield data[:, start:end]
        else: yield data[start:end, :]


theta_f = h5py.File('test_syntax')
g_theta = theta_f.create_dataset('theta', (10, 10), dtype='int32')

g_theta[:, :] = np.eye(10, dtype=np.int32)
li = [[0,1],[1,5],[5,9],[9,10]]

for part in p(g_theta, li, axis=0):
    part = 0

print g_theta[:, :]