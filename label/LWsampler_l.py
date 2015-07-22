# building the parallel frame of the LWsampler

from mpi4py import MPI
from sgd4lda import *
from set_util import load_list, save_data
from sys import stdout
import time
import numpy as np
import h5py
import pstats, cProfile
from label_p import get_code, search

# tag 101: calls the workers to stop
# tag 102: calls the workers to update theta

# tag 111: worker send its current iters
# tag 110: worker send rec
# tag 112: worker send theta
# the num defines the num of updates of the global theta


def lw_frame(num, out_dir, dir, K, V, apprx, train_set_size=20726, doc_per_set=200, alpha=0.01, beta=0.0001,
             batch_size=50, step_size_param=(10**5.2, 10**(-6), 0.33), MH_max=2, word_partition=10000, max_send_times=3):
    """ num is the num_of_samples
        dir: indicates the root folder of each data folder, tmp file folder shall be created in here"""
    fff = stdout.flush
    # ************************************ init params *******************************************************
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    suffix = time.strftime('_%m%d_%H%M%S', time.localtime()) + '_' + str(rank)
    g_name = dir + 'tmp' + suffix + '/' + 'g_theta_file' + suffix + '.h5'
    g_theta = None
    iters = 0
    iters_mean = 0
    H = 1 ** (1 + 0.3) * np.sqrt(size - 1)
    start_time = time.time()
    output_name = out_dir + 'LW_perplexity' + suffix + '.txt'
    sampler = LDSampler(H, dir, rank, train_set_size * doc_per_set, K, V, word_partition * max_send_times, apprx,
                        batch_size=batch_size, alpha=alpha, beta=beta, a=step_size_param[0],
                        b=step_size_param[1], c=step_size_param[2], suffix=suffix)
    if rank != 0:
        rec = np.zeros(V, dtype=bool)
        g_theta_file = h5py.File(g_name, 'w')
        g_theta = g_theta_file.create_dataset('g_theta', (K, V), dtype='float32')

    # init theta and g_theta
    start = 0
    while start < V:
        end = start + word_partition * max_send_times
        end = end * (end <= V) + V * (end > V)

        dummy = sampler.theta[start:end, :]; collect()
        comm.Bcast([dummy, MPI.FLOAT], root=0)
        sampler.theta[start:end, :] = dummy
        if rank != 0: g_theta[start:end, :] = dummy

        start = end

    comm.Bcast([sampler.norm_const, MPI.FLOAT], root=0)

    # ************************************ worker *******************************************************

    if rank != 0:
        # TODO wait for initial perplexity
        # comm.barrier()
        while not comm.Iprobe(source=0, tag=101):
            comm.isend(iters, dest=0, tag=111)

            sampler.update(MH_max, LWsampler=True, g_theta=g_theta, rec=rec)

            if comm.Iprobe(source=0, tag=102):
                comm.recv(source=0, tag=102)

                g_update(comm, sampler.theta, g_theta, sampler.norm_const, K, rec, word_partition * max_send_times)
                g_update(comm, sampler.theta, g_theta, sampler.norm_const, K, rec, word_partition * max_send_times, hash_mode=True)

                rec.fill(0)
                sampler.time_bak = 0

            iters += 1

    # ************************************ master *******************************************************
    else:
        # sche = [2*i**2 for i in xrange(1, num) if 2*i**2 <= num]
        sche = [1, 250, 500, 1000]
        # TODO
        # start_time = get_per_LW(output_name, sampler, start_time, 0)
        # comm.barrier()

        for i in xrange(len(sche)):
            print '0---> update %i of %i' % (i, len(sche))

            while iters_mean < sche[i]:
                iters_mean = get_iters_mean(comm, size)
                print '0---> iter_mean %i' % iters_mean

            # inform to update
            for j in xrange(1, size): comm.isend(None, dest=j, tag=102)

            t_save = sampler.theta[:, :]
            g_recv(comm, sampler.theta, sampler.norm_const, size - 1, K, V, word_partition * max_send_times, apprx)
            start_time = get_per_LW(output_name, sampler, start_time, 0)
            comm.barrier()

            sampler.theta[:, :] = t_save
            code = get_code(t_save)
            comm.Bcast([np.int32(code), MPI.INT], root=0)
            g_recv(comm, sampler.theta, sampler.norm_const, size - 1, K, V, word_partition * max_send_times, apprx)
            start_time = get_per_LW(output_name, sampler, start_time, 0)
            comm.barrier()

        # stop workers, obtain final
        for i in xrange(1, size): comm.send(None, dest=i, tag=101)


def g_update(comm, theta, g_theta, norm_const, K, rec, max_len, send_only=False, hash_mode=False):
    fff = stdout.flush
    map_list = None
    if hash_mode:
        code = np.zeros((K, int(1e5)), dtype=np.int32)
        comm.Bcast([code, MPI.INT], root=0)
        code = code > 0

        code_b = get_code(theta[:, :])
        map_list = search(code, code_b)
    else:
        map_list = range(K)
    comm.Gather([np.int32(rec), MPI.INT], [None, MPI.INT], root=0)
    g_rec = np.int32(rec).copy()
    comm.Bcast([g_rec, MPI.INT], root=0)
    g_rec = g_rec > 0
    comm.Reduce([norm_const, MPI.FLOAT], [None, MPI.FLOAT], op=MPI.SUM, root=0)
    if not send_only: comm.Bcast([norm_const, MPI.FLOAT], root=0)
    true_len = g_rec.sum()
    if true_len <= max_len:

        if rec.sum() != 0:
            theta_batch = theta[:, rec][map_list]
            comm.Send([theta_batch, MPI.FLOAT], dest=0, tag=112)
    else:

        g_mask_list = part_rec(g_rec, max_len)
        mask_list = part_rec(rec, max_len, g_rec=g_rec)

        for i_m in xrange(len(mask_list)):
            if mask_list[i_m].sum() != 0:
                theta_batch = theta[:, mask_list[i_m]][map_list]
                comm.Send([theta_batch, MPI.FLOAT], dest=0, tag=112)

    comm.barrier()


def g_recv(comm, theta, norm_const, num_of_worker, K, V, max_len, apprx, get_only=False):
    fff = stdout.flush
    rec = np.zeros((num_of_worker+1, V), dtype=np.int32)
    comm.Gather([np.zeros(V, dtype=np.int32), MPI.INT], [rec, MPI.INT], root=0)
    rec = rec[1:, :]
    g_rec = rec.sum(0) > 0
    comm.Bcast([np.int32(g_rec), MPI.INT], root=0)
    rec_sum = rec.sum(0)
    times = num_of_worker - rec_sum[rec_sum != 0]
    rec = rec > 0

    comm.Reduce([np.zeros(V, dtype=np.int32), MPI.FLOAT], [norm_const, MPI.FLOAT], op=MPI.SUM, root=0)
    norm_const /= num_of_worker
    if not get_only: comm.Bcast([norm_const, MPI.FLOAT], root=0)

    true_len = g_rec.sum()
    if true_len <= max_len:

        batch_theta = theta[:, g_rec]

        for i in xrange(batch_theta.shape[1]): batch_theta[:, i] *= times[i]
        small_mask = np.zeros(batch_theta.shape[1], dtype=bool)

        for i_n in xrange(num_of_worker):
            if rec[i_n].sum() == 0: continue
            dummy = np.zeros((K, rec[i_n].sum()), dtype=np.float32)
            comm.Recv([dummy, MPI.FLOAT], source=i_n+1, tag=112)
            small_mask.fill(0)
            mask_cnt = 0
            for i_g in xrange(V):
                if g_rec[i_g]:
                    if rec[i_n, i_g]: small_mask[mask_cnt] = True
                    mask_cnt += 1
            batch_theta[:, small_mask] += dummy

        batch_theta /= num_of_worker
        theta[:, g_rec] = batch_theta
    else:

        g_mask_list = part_rec(g_rec, max_len)
        mask_list_list = []
        for i in xrange(num_of_worker):
            mask_list_list += [part_rec(rec[i], max_len, g_rec)]

        cnt_times = 0
        for i_g in xrange(len(g_mask_list)):
            batch_theta = theta[:, g_mask_list[i_g]]; collect()

            for i_t in xrange(batch_theta.shape[1]):
                batch_theta[:, i_t] *= times[i_t + cnt_times]
            cnt_times += batch_theta.shape[1]

            small_mask = np.zeros(batch_theta.shape[1], dtype=bool)
            for i_n in xrange(num_of_worker):
                if mask_list_list[i_n][i_g].sum == 0: continue
                dummy = np.zeros((K, mask_list_list[i_n][i_g].sum()), dtype=np.float32); collect()
                comm.Recv([dummy, MPI.FLOAT], source=i_n+1, tag=112)
                small_mask.fill(0)
                mask_cnt = 0
                for i_v in xrange(V):
                    if g_mask_list[i_g][i_v]:
                        if mask_list_list[i_n][i_g][i_v]: small_mask[mask_cnt] = True
                        mask_cnt += 1
                batch_theta[:, small_mask] += dummy

            batch_theta /= num_of_worker
            theta[:, g_mask_list[i_g]] = batch_theta

def part_rec(rec, max_len, g_rec=None):
    """ rec: bool vec to be part
        max_len: len of each part
        g_rec: if exists, part rec while cnt in g_rec reaches mas_len"""
    mask_list = []

    if g_rec is None:
        part_num = int(np.ceil(rec.sum()/float(max_len)))
        for i in xrange(part_num): mask_list.append(np.zeros(rec.shape[0], dtype=bool))
        cnt = 0
        part_ind = 0
        for i in xrange(rec.shape[0]):
            if rec[i]:
                cnt += 1
                mask_list[part_ind][i] = True

                if cnt == max_len:
                    part_ind += 1
                    cnt = 0
    else:
        part_num = int(np.ceil(g_rec.sum()/float(max_len)))
        for i in xrange(part_num): mask_list.append(np.zeros(rec.shape[0], dtype=bool))
        cnt = 0
        part_ind = 0
        for i in xrange(rec.shape[0]):
            if g_rec[i]:
                cnt += 1
                mask_list[part_ind][i] = rec[i]

                if cnt == max_len:
                    part_ind += 1
                    cnt = 0
    return mask_list


def get_iters_mean(comm, size):
    """ obtaining the iters_mean from all workers"""
    # clean up the old info
    for i in xrange(1, size):
        while comm.Iprobe(source=i, tag=111):
            comm.recv(source=i, tag=111)

    iters = {}

    iters_mean = 0
    flags = [False for _ in xrange(size)]
    out = False
    while not out:
        out = True
        for i in xrange(1, size):
            if comm.Iprobe(source=i, tag=111):
                iters['%i' % i] = comm.recv(source=i, tag=111)
                flags[i] = True
            out = out and flags[i]
    for w in iters:
        iters_mean += iters[w]
    iters_mean /= size-1
    return iters_mean


def get_per_LW(output_name, sampler, start_time, bak_time):
    f = open(output_name, 'a')
    start_time += bak_time
    per_s = time.time()
    print 'computing perplexity: '
    prplx = sampler.get_perp_just_in_time(10)
    print 'perplexity: %.2f' % prplx
    f.write('%.2f\t%.2f\n' % (prplx, per_s - start_time))
    f.close()
    return start_time + time.time() - per_s


if __name__ == '__main__':

    # lw_frame(5, './', '../corpus/b4_ff/', 100, int(1e5))
    lw_frame(1000, '/home/lijm/WORK/yuan/', '/home/lijm/WORK/yuan/b4_ff/', 100, int(1e5), apprx=1)

    # cProfile.runctx("lw_frame(50, '../corpus/b4_ff/', 1000, int(1e5), 2)", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()