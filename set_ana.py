import numpy as np
from pickle import load


mask = np.zeros(100000, dtype=bool)
for i in xrange(20):
    set = load(open('%i/saved_%i' % (0, i), 'r'))
    mask += set[2]

print mask.sum()
print mask[:30000].sum()




