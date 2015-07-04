from mpi4py import MPI
import numpy as np
import time
from scipy import sparse
import h5py
from sys import stdout
import threading
from os import listdir

def func():
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()
    dir = '/home/lijm/WORK/yuan/t_saved/'

    # z = np.array([None for _ in xrange(10000)], dtype=object)
    # for i in xrange(10000):
    #     z[i] = np.ones(100, dtype=np.int32)
    #
    # a = sparse.csr_matrix(np.zeros((1000, int(1e5)), dtype=np.int32))
    #
    # for i in xrange(10000):
    #     x = np.random.randint(a.shape[0])
    #     y = np.random.randint(a.shape[1])
    #     a[x, y] = 1

    start = time.time()
    for i in xrange(10):
        for file_name in listdir(dir+'1/'):
            if 'test' in file_name: continue
            np.load(dir+'1/' + file_name).tolist()

    # for i in xrange(100):
    #     np.save(dir+'test_zzz', z)
    # print time.time() - start
    #
    # start = time.time()
    # for i in xrange(100):
    #     np.load(dir+'test_zzz'+'.npy')
    # print time.time() - start
    #
    # start = time.time()
    # for i in xrange(100):
    #     a.data.dump(dir + 'a_data')
    #     a.indices.dump(dir + 'a_indices')
    #     a.indptr.dump(dir + 'a_indptr')
    # print time.time() - start
    #
    # start = time.time()
    # for i in xrange(100):
    #     sparse.csr_matrix((np.load(dir + 'a_data'), np.load(dir + 'a_indices'), np.load(dir + 'a_indptr')),
    #                       shape=(1000, int(1e5)))
    print time.time() - start


if __name__ == '__main__':
    func()