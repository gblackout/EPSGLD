from mpi4py import MPI
from sgd4lda import *
from sys import stdout
import time
import numpy as np
import h5py

# tag def
M_REC = 100
M_THETA = 101
W_REC = 110
W_THETA = 111

# const
NFLT = np.float32
NINT = np.int32
NDBL = np.float64


def run_DSGLD(num, out_dir, dir, K, V, traject, apprx, train_set_size=20726, doc_per_set=200, alpha=0.01, beta=0.0001, batch_size=50,
              step_size_param=(0.01, 1000, 0.55), MH_max=10, word_partition=10000, max_send_times=3):
    # ************************************ init params *******************************************************
    fff = stdout.flush
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    suffix = time.strftime('_%m%d_%H%M%S', time.localtime()) + '_' + str(rank)
    output_name = out_dir + 'DSGLD_perplexity' + suffix + '.txt'
    tmp_dir = dir + 'tmp' + suffix + '/'
    trans_time = 9 * apprx * K * V / 1e9

    part_list = []
    start = 0
    while start < V:
        end = start + word_partition*max_send_times
        end = end * (end <= V) + V * (end > V)
        part_list.append([start, end])
        start = end

    sampler = LDSampler(0, dir, rank, train_set_size * doc_per_set, K, V, word_partition * max_send_times, apprx,
                        batch_size=batch_size, alpha=alpha, beta=beta, epsilon=step_size_param[0],
                        tau0=step_size_param[1], kappa=step_size_param[2], suffix=suffix)

    # ************************************ worker *******************************************************
    start_time = time.time()
    if rank != 0:

        comm.reduce(sampler.get_perp_just_in_time(10), op=MPI.SUM, root=0)
        comm.barrier()

        work_time = time.time()
        for iter in xrange(num):
            print 'rank %i iter: %i' % (rank, iter); fff()
            sampler.update(MH_max)

            if (iter + 1) % traject == 0:
                comm.Gather(np.float32(time.time() - work_time), None, root=0)
                comm.Gather(np.float32(sampler.time_bak), None, root=0)

                comm.reduce(sampler.get_perp_just_in_time(10), op=MPI.SUM, root=0)
                comm.barrier()

                g_update(comm, sampler.theta, sampler.norm_const, K, part_list)
                sampler.time_bak = 0; work_time = time.time()

    # ************************************ master *******************************************************
    else:
        theta_f = h5py.File(tmp_dir + 'theta_g' + '.h5', 'w')
        g_theta = theta_f.create_dataset('theta', (K, V), dtype='float32')

        io_time_list = np.zeros(size, dtype=np.float32)
        work_time_list = np.zeros(size, dtype=np.float32)

        start_time = get_per_DSGLD(output_name, comm, start_time, 0)
        time_bak = 0
        for iter in xrange(num):
            if (iter + 1) % traject == 0:
                print '*********************************************************'
                print '***************************** iter %i ********************' % (iter - traject + 1)
                print '*********************************************************'
                work_time_list.fill(0); io_time_list.fill(0)
                comm.Gather(np.float32(0), [work_time_list, MPI.FLOAT], root=0)
                comm.Gather(np.float32(0), [io_time_list, MPI.FLOAT], root=0)
                print 'plus io_time: current: %.2f  start: %.2f  bak: %.2f  io: %.2f' % (time.time(), start_time, time_bak, work_time_list.max() - (work_time_list - io_time_list).max())
                fff()
                time_bak += work_time_list.max() - (work_time_list - io_time_list).max()

                start_time = get_per_DSGLD(output_name, comm, start_time, time_bak); time_bak = 0

                recv_time = time.time()
                g_recv(comm, sampler.theta, g_theta, size - 1, K, part_list)
                print 'plus recv_time: current: %.2f  start: %.2f  bak: %.2f  recv: %.2f elapsed_time %.2f' % (time.time(), start_time, time_bak, time.time() - recv_time - trans_time, time.time() - recv_time)
                fff()
                if time.time() - recv_time - trans_time > 0: time_bak += time.time() - recv_time - trans_time


def g_update(comm, theta, norm_const, K, part_list):
    """ assume mem is large enough"""
    fff = stdout.flush

    # ======================================= send to master ================================================
    send_np(comm, NFLT, norm_const, dest=0, tag=W_THETA)
    for start, end in part_list:
        send_np(comm, NFLT, theta[:, start:end], dest=0, tag=W_THETA)

    # ======================================= recv from master ================================================
    recv_np(comm, NFLT, buf=norm_const, source=0, tag=M_THETA)
    for start, end in part_list:
        theta[:, start:end] = recv_np(comm, NFLT, xy=[K, end-start], source=0, tag=M_THETA)

    comm.barrier()


def g_recv(comm, theta, g_theta, num_of_worker, K, part_list):

    # ======================================= recv last ================================================
    g_norm_const = recv_np(comm, NFLT, xy=[K, 1], source=num_of_worker, tag=W_THETA)
    for start, end in part_list:
        g_theta[:, start:end] = recv_np(comm, NFLT, xy=[K, end-start], source=num_of_worker, tag=W_THETA)

    # ======================================= recv&send ================================================
    for i in xrange(num_of_worker):

        if i != num_of_worker - 1:
            norm_const = recv_np(comm, NFLT, xy=[K, 1], source=i + 1, tag=W_THETA)
            for start, end in part_list:
                theta[:, start:end] = recv_np(comm, NFLT, xy=[K, end-start], source=i + 1, tag=W_THETA)

        send_np(comm, NFLT, g_norm_const, dest=i + 1, tag=M_THETA)
        for start, end in part_list:
            send_np(comm, NFLT, g_theta[:, start:end], dest=i + 1, tag=M_THETA)

        # ------------------------------------- swap -------------------------------------------------
        tt = theta; theta = g_theta; g_theta = tt
        tt = norm_const; norm_const = g_norm_const; g_norm_const = tt

    comm.barrier()


def g_coupling(comm, g_theta, K, part_list, group_list):
    fff = stdout.flush

    for group in group_list:
        # ======================================= recv from each group ================================================
        norm_const = np_float(K, 1)
        for start, end in part_list:
            g_theta[:, start:end] = 0

        mem_num = len(group)
        for i in group:
            norm_const += recv_np(comm, NFLT, xy=[K, 1], source=i, tag=W_THETA)
            for start, end in part_list:
                g_theta[:, start:end] += recv_np(comm, NFLT, xy=[K, end-start], source=i, tag=W_THETA) / mem_num

        norm_const /= mem_num

        # ======================================= send to each group ================================================
        for i in group:
            send_np(comm, NFLT, norm_const, dest=i, tag=M_THETA)
            for start, end in part_list:
                send_np(comm, NFLT, g_theta[:, start:end], dest=i, tag=M_THETA)



def send_np(comm, type, buf, **kwargs):
    comm.Send([buf, n2m(type)], **kwargs)


def recv_np(comm, type, **kwargs):
    """
    you can input your own buff using buf=abc
    or using xy and we will init one for you
    """

    if 'xy' in kwargs:
        tmp = np.zeros((kwargs['xy'][0], kwargs['xy'][1]), dtype=type)
    elif 'buf' in kwargs:
        tmp = kwargs['buf']
    else:
        print ''
        raise ValueError('please give me either xy or buf')

    comm.Recv([tmp, n2m(type)], **kwargs)

    return tmp


def n2m(type):
    try:
        return {
            np.int32: MPI.INT,
            np.float32: MPI.FLOAT,
            np.float64: MPI.DOUBLE
        }[type]
    except:
        print 'type not recorded'


def p(data, li, axis=1):
    for start, end in li:
        if axis: yield data[:, start:end]
        else: yield data[start:end, :]


def np_float(x, y=None):
    if y is None: return np.zeros(x, dtype=np.float32)
    return np.zeros((x, y), dtype=np.float32)


def np_int(x, y=None):
    if y is None: return np.zeros(x, dtype=np.int32)
    return np.zeros((x, y), dtype=np.int32)


def get_per_DSGLD(output_name, comm, start_time, bak_time):
    print 'computing perplexity: '

    # debug
    print 'get_per_DSGLD: current: %.2f  start: %.2f  bak: %.2f' % (time.time(), start_time, bak_time)
    stdout.flush()

    start_time += bak_time
    per_s = time.time()

    prplx = comm.reduce(0, op=MPI.SUM, root=0) / (comm.Get_size() - 1)
    print 'perplexity: %.2f' % prplx
    stdout.flush()

    f = open(output_name, 'a')
    f.write('%.2f\t%.2f\n' % (prplx, per_s - start_time))
    f.close()
    comm.barrier()

    print 'get_per_DSGLD: elapsed_time: %.2f' % (time.time() - per_s)
    stdout.flush()

    return start_time + time.time() - per_s


if __name__ == '__main__':

    # run_DSGLD(25, './', '../corpus/b4_ff/', 100, int(1e5), traject=5, apprx=1)

    run_DSGLD(1000, '/home/lijm/WORK/yuan/', '/home/lijm/WORK/yuan/b4_ff/', 1000, int(1e5), traject=5, apprx=1)