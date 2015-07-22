from label_p import *
import h5py
from sgd4lda import LDSampler
import time

data_dir = '/home/lijm/WORK/yuan/b4_ff/tmp_0718_215239_'

print 'init one'
theta = h5py.File(data_dir+str(2), 'r')['theta'][:, :]

for i in xrange(3, 7):
    print 'init', i
    a = h5py.File(data_dir+str(i), 'r')
    theta += a['theta'][:, :]

theta /= 5


print 'init sampler '
num = 12000
train_set_size = 20726
rank = 1
doc_per_set = 200
V = int(1e5)
K = 1000
dir = '/home/lijm/WORK/yuan/b4_ff/'
out_dir = '/home/lijm/WORK/yuan/'
max_len = 10000
output_name = out_dir + 'serial_perplexity' + time.strftime('_%m%d_%H%M%S', time.localtime()) + '.txt'

sampler = LDSampler(0, dir, 1, train_set_size*doc_per_set, K, V, max_len, 1)


print 'init theta/const'
sampler.theta[:, :] = theta
sampler.norm_const = np.sum(theta, 1)[:, np.newaxis]

print 'compute prplx'
print sampler.get_perp_just_in_time(10)



print 'init one'
theta = h5py.File(data_dir+str(2), 'r')['theta'][:, :]
code = get_code(theta)

for i in xrange(3, 7):
    print 'init', i
    a = h5py.File(data_dir+str(i), 'r')
    tmp = a['theta'][:, :]
    theta += tmp[search(code, get_code(tmp))]

theta /= 5

print 'init theta/const'
sampler.theta[:, :] = theta
sampler.norm_const = np.sum(theta, 1)[:, np.newaxis]

print 'compute prplx'
print sampler.get_perp_just_in_time(10)