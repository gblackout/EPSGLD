from sgd4lda import LDSampler, slice_list, get_per
import itertools
from mpi4py import MPI
import time


def run():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num = 3000
    train_set_size = 20726
    doc_per_set = 200
    V = int(1e5)
    K = 1000
    jump = 100
    jump_bias = 0
    jump_hold = 0
    # dir = '../corpus/b4_ff/'
    # out_dir = './'
    dir = '/home/lijm/WORK/yuan/b4_ff/'
    out_dir = '/home/lijm/WORK/yuan/'
    max_len = 10000

    a_list = [(100000*(10**0.2)**x, x) for x in xrange(0, 11)]
    b_list = [(0.0001*(10**(-0.2))**x, x) for x in xrange(0, 11)]

    work_list = slice_list(list(itertools.product(a_list, b_list)), size)

    for a, b in work_list[rank]:
        suffix = time.strftime('_%m%d_%H%M%S', time.localtime()) + '_' + str(rank)
        output_name = out_dir + 'search_%i_%i' % (a[1], b[1]) + suffix + '.txt'

        sampler = LDSampler(0, dir, 1, train_set_size*doc_per_set, K, V, max_len, 1,
                            a=a[0], b=b[0], c=0.33, suffix=suffix)

        start_time = time.time()
        for i in xrange(num):
            print 'iter--->', i
            sampler.update(10)

            if i < jump_bias and i != 0:
                start_time = get_per(output_name, sampler, start_time)
            elif (i + 1) % jump == 0 and (i + 1) >= jump_hold:
                start_time = get_per(output_name, sampler, start_time)


def run_serial():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num = 15000
    train_set_size = 20726
    doc_per_set = 200
    V = int(1e5)
    K = 1000
    jump = 100
    jump_bias = 0
    jump_hold = 0
    # dir = '../corpus/b4_ff/'
    # out_dir = './'
    dir = '/home/lijm/WORK/yuan/b4_ff/'
    out_dir = '/home/lijm/WORK/yuan/'
    max_len = 10000


    suffix = time.strftime('_%m%d_%H%M%S', time.localtime()) + '_' + str(rank)
    output_name = out_dir + 'serial_%i' % rank + suffix + '.txt'

    sampler = LDSampler(0, dir, 1, train_set_size*doc_per_set, K, V, max_len, 1, suffix=suffix)

    start_time = time.time()
    for i in xrange(num):
        print 'iter--->', i
        sampler.update(2)

        if i < jump_bias and i != 0:
            start_time = get_per(output_name, sampler, start_time)
        elif (i + 1) % jump == 0 and (i + 1) >= jump_hold:
            start_time = get_per(output_name, sampler, start_time)


if __name__ == '__main__':
    run_serial()