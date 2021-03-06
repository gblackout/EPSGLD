# building the parallel frame of the LWsampler

from mpi4py import MPI
from sgd4lda import *
from set_util import load_list, save_data
from sys import stdout
import time
import numpy as np
import h5py
import pstats, cProfile

# tag 101: calls the workers to stop
# tag 102: calls the workers to update theta

# tag 111: worker send its current iters
# tag 110: worker send rec
# tag 112: worker send theta
# the num defines the num of updates of the global theta


def lw_frame(num, out_dir, dir, K, V, apprx, train_set_size=20726, doc_per_set=200, alpha=0.01, beta=0.0001,
             batch_size=50, step_size_param=(10**5.2, 10**(-6), 0.33), MH_max=2, word_partition=10000, max_send_times=3):
    """
        the main framework of embarrassingly parallel implemented using MPI: for 10708 prj, you shouldn't worry too
        much about if you don't know, there is only a small fraction of code has things to do with the distributed
        algorithm (the pseudo code I posted on my paper), others are not so important since Petuum is going to take
        care of it

        input:
            num: the number of samples
            dir: indicates the root folder of each data folder, tmp file folder shall be created in here
    """
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
    start_time = 0
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

        cul_time = time.time()
        dummy = sampler.theta[start:end, :]; collect()
        x = time.time()
        comm.Bcast([dummy, MPI.FLOAT], root=0)
        x = time.time() - x
        sampler.theta[start:end, :] = dummy
        start_time += time.time() - cul_time - 30 * (dummy.shape[0]*dummy.shape[1]) / 1e9 - x

        if rank != 0: g_theta[start:end, :] = dummy

        start = end
        dummy = None; collect()
    comm.Bcast([sampler.norm_const, MPI.FLOAT], root=0)

    # ************************************ worker *******************************************************
    # for 10708 prj: the worker essentially do  nothing but update their model locally, and updates its local copy of
    # global theta (I refer to theta in my paper) using g_update (ignore its implementation since we now use Petuum for
    # synchronization)

    if rank != 0:
        # wait for initial perplexity
        comm.barrier()
        work_time = time.time()
        while not comm.Iprobe(source=0, tag=101):
            comm.isend(iters, dest=0, tag=111)

            sampler.update(MH_max, LWsampler=True, g_theta=g_theta, rec=rec)

            if comm.Iprobe(source=0, tag=102):
                comm.recv(source=0, tag=102)

                comm.Gather(np.float32(time.time()-work_time), None, root=0)
                comm.Gather(np.float32(sampler.time_bak), None, root=0)
                g_update(comm, sampler.theta, g_theta, sampler.norm_const, K, rec, word_partition * max_send_times)
                rec.fill(0)
                comm.barrier()
                sampler.time_bak = 0; work_time = time.time()

            iters += 1

        comm.Gather(np.float32(time.time() - work_time), None, root=0)
        comm.Gather(np.float32(sampler.time_bak), None, root=0)
        g_update(comm, sampler.theta, g_theta, sampler.norm_const, K, rec, word_partition * max_send_times,
                 send_only=True)

    # ************************************ master *******************************************************
    # for 10708 prj: master essentially does two things: update the global theta using
    # global theta = (theta_1 + .. + theta_n) / n where n is the number of workers; and push this back to worker
    # I think you can ignore the code below and build them from scratch using Petuum, which should be far more simpler
    # than mine
    else:
        sche = [2*i**2 for i in xrange(1, num) if 2*i**2 <= num]
        io_time_list = np.zeros(size, dtype=np.float32)
        work_time_list = np.zeros(size, dtype=np.float32)
        start_time = time.time() - start_time
        start_time = get_per_LW(output_name, sampler, start_time, 0)
        comm.barrier()

        for i in xrange(len(sche)):
            print '0---> update %i of %i' % (i, len(sche))

            while iters_mean < sche[i]:
                iters_mean = get_iters_mean(comm, size)
                print '0---> iter_mean %i' % iters_mean

            # inform to update
            for j in xrange(1, size): comm.isend(None, dest=j, tag=102)
            comm.Gather(np.float32(0), [work_time_list, MPI.FLOAT], root=0)
            comm.Gather(np.float32(0), [io_time_list, MPI.FLOAT], root=0)
            recv_time = time.time()
            trans_time = g_recv(comm, sampler.theta, sampler.norm_const, size - 1, K, V, word_partition * max_send_times, apprx)
            time_bak = time.time() - recv_time - trans_time + work_time_list.max() - (work_time_list - io_time_list).max()
            start_time = get_per_LW(output_name, sampler, start_time, time_bak)
            work_time_list.fill(0); io_time_list.fill(0)
            comm.barrier()

        # stop workers, obtain final
        for i in xrange(1, size): comm.send(None, dest=i, tag=101)
        comm.Gather(np.float32(0), [work_time_list, MPI.FLOAT], root=0)
        comm.Gather(np.float32(0), [io_time_list, MPI.FLOAT], root=0)
        recv_time = time.time()
        trans_time = g_recv(comm, sampler.theta, sampler.norm_const, size - 1, K, V, word_partition * max_send_times, apprx, get_only=True)
        time_bak = time.time() - recv_time - trans_time + work_time_list.max() - (work_time_list - io_time_list).max()
        get_per_LW(output_name, sampler, start_time, time_bak)


def g_update(comm, theta, g_theta, norm_const, K, rec, max_len, send_only=False):
    fff = stdout.flush
    comm.Gather([np.int32(rec), MPI.INT], [None, MPI.INT], root=0)
    g_rec = np.int32(rec).copy()
    comm.Bcast([g_rec, MPI.INT], root=0)
    g_rec = g_rec > 0
    comm.Reduce([norm_const, MPI.FLOAT], [None, MPI.FLOAT], op=MPI.SUM, root=0)
    if not send_only: comm.Bcast([norm_const, MPI.FLOAT], root=0)

    true_len = g_rec.sum()
    if true_len <= max_len:
        if rec.sum() != 0:
            theta_batch = theta[:, rec]; collect()
            comm.Send([theta_batch, MPI.FLOAT], dest=0, tag=112)
            theta_batch = None; collect()
        if send_only: return
        theta_batch = np.zeros((K, true_len), dtype=np.float32); collect()
        comm.Bcast([theta_batch, MPI.FLOAT], root=0)
        theta[:, g_rec] = theta_batch
        g_theta[:, g_rec] = theta_batch
        theta_batch = None; collect()
    else:
        g_mask_list = part_rec(g_rec, max_len)
        mask_list = part_rec(rec, max_len, g_rec=g_rec)

        for i_m in xrange(len(mask_list)):
            if mask_list[i_m].sum() != 0:
                theta_batch = theta[:, mask_list[i_m]]; collect()
                comm.Send([theta_batch, MPI.FLOAT], dest=0, tag=112)
                theta_batch = None; collect()
            if send_only: continue
            theta_batch = np.zeros((K, g_mask_list[i_m].sum()), dtype=np.float32); collect()
            comm.Bcast([theta_batch, MPI.FLOAT], root=0)
            theta[:, g_mask_list[i_m]] = theta_batch
            g_theta[:, g_mask_list[i_m]] = theta_batch
            theta_batch = None; collect()


def g_recv(comm, theta, norm_const, num_of_worker, K, V, max_len, apprx, get_only=False):
    trans_time = time.time()
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
    trans_time = time.time() - trans_time

    true_len = g_rec.sum()
    if true_len <= max_len:
        batch_theta = theta[:, g_rec]; collect()

        cul_time = time.time()
        for i in xrange(batch_theta.shape[1]): batch_theta[:, i] *= times[i]
        small_mask = np.zeros(batch_theta.shape[1], dtype=bool)
        trans_time += time.time() - cul_time + 1.5 * apprx * batch_theta.shape[0] * batch_theta.shape[1] / 1e9

        for i_n in xrange(num_of_worker):
            if rec[i_n].sum() == 0: continue
            cul_time = time.time()
            dummy = np.zeros((K, rec[i_n].sum()), dtype=np.float32)
            trans_time += time.time() - cul_time
            comm.Recv([dummy, MPI.FLOAT], source=i_n+1, tag=112)

            cul_time = time.time()
            small_mask.fill(0)
            mask_cnt = 0
            for i_g in xrange(V):
                if g_rec[i_g]:
                    if rec[i_n, i_g]: small_mask[mask_cnt] = True
                    mask_cnt += 1

            batch_theta[:, small_mask] += dummy
            trans_time += time.time() - cul_time + 3 * apprx * dummy.shape[0] * dummy.shape[1] / 1e9

        cul_time = time.time()
        batch_theta /= num_of_worker
        trans_time += time.time() - cul_time

        trans_time += (3 * num_of_worker + 1.5) * apprx * batch_theta.shape[0] * batch_theta.shape[1] / 1e9
        if not get_only: comm.Bcast([batch_theta, MPI.FLOAT], root=0)
        theta[:, g_rec] = batch_theta
        return trans_time

    else:
        cul_time = time.time()
        g_mask_list = part_rec(g_rec, max_len)
        mask_list_list = []
        for i in xrange(num_of_worker):
            mask_list_list += [part_rec(rec[i], max_len, g_rec)]
        trans_time += time.time() - cul_time

        cnt_times = 0
        for i_g in xrange(len(g_mask_list)):
            batch_theta = theta[:, g_mask_list[i_g]]; collect()

            cul_time = time.time()
            for i_t in xrange(batch_theta.shape[1]):
                batch_theta[:, i_t] *= times[i_t + cnt_times]
            cnt_times += batch_theta.shape[1]
            trans_time += time.time() - cul_time + 1.5 * apprx * batch_theta.shape[0] * batch_theta.shape[1] / 1e9

            small_mask = np.zeros(batch_theta.shape[1], dtype=bool)
            for i_n in xrange(num_of_worker):
                if mask_list_list[i_n][i_g].sum == 0: continue
                cul_time = time.time()
                dummy = np.zeros((K, mask_list_list[i_n][i_g].sum()), dtype=np.float32); collect()
                trans_time += time.time() - cul_time
                comm.Recv([dummy, MPI.FLOAT], source=i_n+1, tag=112)

                cul_time = time.time()
                small_mask.fill(0)
                mask_cnt = 0
                for i_v in xrange(V):
                    if g_mask_list[i_g][i_v]:
                        if mask_list_list[i_n][i_g][i_v]: small_mask[mask_cnt] = True
                        mask_cnt += 1

                batch_theta[:, small_mask] += dummy
                trans_time += time.time() - cul_time + 3 * apprx * dummy.shape[0] * dummy.shape[1] / 1e9

            cul_time = time.time()
            batch_theta /= num_of_worker
            trans_time += time.time() - cul_time

            trans_time += (3 * num_of_worker + 1.5) * apprx * batch_theta.shape[0] * batch_theta.shape[1] / 1e9
            if not get_only: comm.Bcast([batch_theta, MPI.FLOAT], root=0)
            theta[:, g_mask_list[i_g]] = batch_theta

        return trans_time


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

    # lw_frame(1000, './', '../corpus/b4_ff/', 100, int(1e5), 1)
    # lw_frame(20, '/home/lijm/WORK/yuan/', '/home/lijm/WORK/yuan/b4_ff/', 100, int(1e5), apprx=1)

    # cProfile.runctx("lw_frame(50, '../corpus/b4_ff/', 1000, int(1e5), 2)", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--mode", dest="MODE", type="string", default='get', help="")
    parser.add_option("--k", dest="K", type="int", default=1000, help="")
    parser.add_option("--steps", dest="STEPS", type="int", default=30000, help="")
    (options, args) = parser.parse_args()

    if options.MODE == 'wiki':
        lw_frame(options.STEPS, '/home/lijm/WORK/yuan/', '/home/lijm/WORK/yuan/b4_ff/', options.K, int(1e5), apprx=1)
    elif options.MODE == 'clueweb':
        lw_frame(options.STEPS, '/home/lijm/WORK/yuan/', '/home/lijm/WORK/yuan/clueweb/sgld_data/', options.K, int(1e5), apprx=1)