

import util_funcs as u
import numpy as np

f = open('vec.txt','r')
phi = []
for line in f:
    phi += [float(line)]

phi1 = np.array([phi[i] for i in xrange(len(phi)) if i % 2 == 0], dtype=np.float32)
phi = np.zeros((1000, 1), dtype=np.float32)
phi[:, 0] = phi1

table_l = np.zeros((1, 1000), dtype=np.int32)
table_h = np.zeros((1, 1000), dtype=np.int32)
table_p = np.zeros((1, 1000), dtype=np.float32)
mask = np.ones(1000, dtype=bool)

u.gen_alias_table_light(table_h, table_l, table_p, phi, mask)
