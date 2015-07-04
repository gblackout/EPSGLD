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

    for start, end in part_list:
        comm.Send([theta[:, start:end], MPI.FLOAT], dest=0, tag=W_THETA)
    comm.Send([norm_const, MPI.FLOAT], dest=0, tag=W_THETA)

    for start, end in part_list:
        theta_batch = np.zeros((K, end - start), dtype=np.float32)
        comm.Recv([theta_batch, MPI.FLOAT], source=0, tag=M_THETA)
        theta[:, start:end] = theta_batch
        theta_batch = None; collect()
    comm.Recv([norm_const, MPI.FLOAT], source=0, tag=M_THETA)
    comm.barrier()


def g_recv(comm, theta, g_theta, num_of_worker, K, part_list):
    fff = stdout.flush
    norm_const = np.zeros((K, 1), dtype=np.float32)
    g_norm_const = np.zeros((K, 1), dtype=np.float32)

    for start, end in part_list:
        theta_batch = np.zeros((K, end - start), dtype=np.float32)
        comm.Recv([theta_batch, MPI.FLOAT], source=num_of_worker, tag=W_THETA)
        g_theta[:, start:end] = theta_batch
        theta_batch = None; collect()
    comm.Recv([g_norm_const, MPI.FLOAT], source=num_of_worker, tag=W_THETA)

    for i in xrange(num_of_worker):
        if i != num_of_worker - 1:
            for start, end in part_list:
                theta_batch = np.zeros((K, end - start), dtype=np.float32)
                comm.Recv([theta_batch, MPI.FLOAT], source=i + 1, tag=W_THETA)
                theta[:, start:end] = theta_batch
                theta_batch = None; collect()
            comm.Recv([norm_const, MPI.FLOAT], source=i + 1, tag=W_THETA)

        for start, end in part_list:
            comm.Send([g_theta[:, start:end], MPI.FLOAT], dest=i + 1, tag=M_THETA)
        comm.Send([g_norm_const, MPI.FLOAT], dest=i + 1, tag=M_THETA)

        tt = theta; theta = g_theta; g_theta = tt
        tt = norm_const; norm_const = g_norm_const; g_norm_const = tt
    comm.barrier()

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