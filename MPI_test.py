from mpi4py import MPI
import numpy as np
import time
from scipy import sparse
import h5py
from sys import stdout
import threading
from os import listdir

def func():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    a = np.ones(3, dtype=np.int32) * rank

    if rank == 0:
        status = MPI.Status()
        cnt = 0
        while cnt < size - 1:
            if comm.Iprobe(source=MPI.ANY_SOURCE, status=status):
                src = status.Get_source(); print src
                comm.Recv([a, MPI.INT], source=src)
                print a
                cnt += 1
    else:
        comm.Send([a, MPI.INT], dest=0)



if __name__ == '__main__':
    func()