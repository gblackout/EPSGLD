# building the parallel frame of the LWsampler

from mpi4py import MPI
from sgd4lda import *
from set_util import load_list, save_data
from sys import stdout
import time
import numpy as np
import h5py
import pstats, cProfile
from DSGLD import mk_plist, NFLT, NINT, n2m, send_np, recv_np, np_float, np_int
from timer import Timer

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
    # ==================================================================================================================
    # init model
    # ==================================================================================================================
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    iters_mean = 0
    b_timer = Timer()
    max_len = word_partition * max_send_times
    H = 1 ** (1 + 0.3) * np.sqrt(size - 1)
    part_list = mk_plist(max_len, V)
    sche = [2*i**2 for i in xrange(1, num) if 2*i**2 <= num]
    suffix = time.strftime('_%m%d_%H%M%S', time.localtime()) + '_' + str(rank)
    output_name = out_dir + 'LW_perplexity' + suffix + '.txt'

    sampler = LDSampler(H, dir, rank, train_set_size * doc_per_set, K, V, max_len, apprx,
                        batch_size=batch_size, alpha=alpha, beta=beta, a=step_size_param[0],
                        b=step_size_param[1], c=step_size_param[2], suffix=suffix)

    if rank != 0:
        worker_ps(comm, sampler, K, V, MH_max, max_len, suffix, dir)

    # init theta and g_theta
    start_time = time.time()
    for start, end in part_list:
        b_timer.go(); dummy = sampler.theta[start:end, :]
        bcast_np(comm, NFLT, False, buf=dummy, root=0); b_timer.stop()
        start_time += b_timer() - est_time(True, dummy.shape, apprx)

    comm.Bcast([sampler.norm_const, MPI.FLOAT], root=0)

    # ==================================================================================================================
    # working
    # ==================================================================================================================
    start_time = get_per_LW(output_name, sampler, start_time, 0)
    comm.barrier()

    for i in xrange(len(sche)):
        print '0---> update %i of %i' % (i, len(sche))

        while iters_mean < sche[i]:
            iters_mean = get_iters_mean(comm, size)
            print '0---> iter_mean %i' % iters_mean

        # inform to update
        for j in xrange(1, size): comm.isend(None, dest=j, tag=102)
        time_bak = g_recv(comm, size, sampler.theta, sampler.norm_const, size - 1, K, V, max_len, apprx)

        start_time = get_per_LW(output_name, sampler, start_time, time_bak)
        comm.barrier()

    for i in xrange(1, size): comm.send(None, dest=i, tag=101)
    time_bak = g_recv(comm, size, sampler.theta, sampler.norm_const, size - 1, K, V, max_len, apprx, get_only=True)

    get_per_LW(output_name, sampler, start_time, time_bak)
    comm.barrier()


def g_update(comm, theta, g_theta, norm_const, K, rec, max_len, send_only=False):
    fff = stdout.flush
    # ==================================================================================================================
    # sync rec and norm_const
    # ==================================================================================================================
    comm.Gather([np.int32(rec), MPI.INT], [None, MPI.INT], root=0)
    g_rec = bcast_np(comm, NINT, True, root=0, buf=np.int32(rec).copy()) > 0

    reduce_np(comm, NFLT, True, buf=norm_const, op=MPI.SUM, root=0)
    if not send_only: bcast_np(comm, NFLT, False, buf=norm_const, root=0)

    # ==================================================================================================================
    # sync data
    # ==================================================================================================================
    true_len = g_rec.sum()

    if true_len <= max_len:
        # ================================================one run=======================================================
        if rec.sum() != 0:
            send_np(comm, NFLT, theta[:, rec], dest=0, tag=112)

        if not send_only:
            buf = bcast_np(comm, NFLT, True, xy=[K, true_len], root=0)
            theta[:, g_rec] = buf
            g_theta[:, g_rec] = buf

    else:
        # ================================================multiple run==================================================
        g_mask_list = part_rec(g_rec, max_len)
        mask_list = part_rec(rec, max_len, g_rec=g_rec)

        for i_m in xrange(len(mask_list)):

            if mask_list[i_m].sum() != 0:
                send_np(comm, NFLT, theta[:, mask_list[i_m]], dest=0, tag=112)

            if not send_only:
                buf = bcast_np(comm, NFLT, True, xy=[K, g_mask_list[i_m].sum()], root=0)
                theta[:, g_mask_list[i_m]] = buf
                g_theta[:, g_mask_list[i_m]] = buf

    # wait for prplx
    comm.barrier()


def g_recv(comm, size, theta, norm_const, num_of_worker, K, V, max_len, apprx, get_only=False):
    fff = stdout.flush
    status = MPI.Status()
    io_time_list = np.zeros(size, dtype=np.float32)
    work_time_list = np.zeros(size, dtype=np.float32)
    t_timer = Timer()
    # ==================================================================================================================
    # sync rec and norm_const
    # ==================================================================================================================

    comm.Gather(np.float32(0), [work_time_list, MPI.FLOAT], root=0)
    comm.Gather(np.float32(0), [io_time_list, MPI.FLOAT], root=0)

    rec = gather_np(comm, NINT, False, xy=[num_of_worker+1, V], root=0) > 0
    g_rec = rec.sum(0)
    comm.Bcast([g_rec, MPI.INT], root=0)
    times = num_of_worker - g_rec[g_rec != 0]
    g_rec = g_rec > 0

    norm_const = reduce_np(comm, NFLT, False, buf=norm_const, op=MPI.SUM, root=0) / num_of_worker
    if not get_only: comm.Bcast([norm_const, MPI.FLOAT], root=0)

    # ==================================================================================================================
    # sync data
    # ==================================================================================================================
    true_len = g_rec.sum()

    if true_len <= max_len:
        # ================================================one run=======================================================
        t_timer.go(); batch_theta = theta[:, g_rec]; t_timer.stop()
        batch_theta *= times

        recv_cnt = 0; t_timer.go()
        while recv_cnt <  num_of_worker:
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=112, status=status):
                src = status.Get_source(); t_timer.stop()
                local_len = rec[src - 1].sum()
                if local_len != 0:
                    buf = recv_np(comm, NFLT, xy=[K, local_len], source=src, tag=112)
                    batch_theta[:, mk_mask(true_len, g_rec, rec[src - 1])] += buf
                recv_cnt += 1; t_timer.go()
        t_timer.stop()

        batch_theta /= num_of_worker

        if not get_only: comm.Bcast([batch_theta, MPI.FLOAT], root=0)
        t_timer.go(); theta[:, g_rec] = batch_theta; t_timer.stop()

    else:
        # ================================================multiple run==================================================
        g_mask_list = part_rec(g_rec, max_len)
        mask_list_list = []
        for i in xrange(num_of_worker):
            mask_list_list += [part_rec(rec[i], max_len, g_rec)]

        cnt_times = 0
        for i_g in xrange(len(g_mask_list)):
            t_timer.go(); batch_theta = theta[:, g_mask_list[i_g]]; t_timer.stop()
            g_len = batch_theta.shape[1]
            batch_theta *= times[cnt_times: cnt_times + g_len]
            cnt_times += g_len

            recv_cnt = 0; t_timer.go()
            while recv_cnt <  num_of_worker:
                if comm.Iprobe(source=MPI.ANY_SOURCE, tag=112, status=status):
                    src = status.Get_source(); t_timer.stop()
                    local_len = mask_list_list[src-1][i_g].sum()
                    if local_len != 0:
                        buf = recv_np(comm, NFLT, xy=[K, local_len], source=src, tag=112)
                        batch_theta[:, mk_mask(g_len, g_mask_list[i_g], mask_list_list[src-1][i_g])] += buf
                    recv_cnt += 1; t_timer.go()
            t_timer.stop()

            batch_theta /= num_of_worker

            if not get_only: comm.Bcast([batch_theta, MPI.FLOAT], root=0)
            t_timer.go(); theta[:, g_mask_list[i_g]] = batch_theta; t_timer.stop()

    return t_timer() - est_time(True, [K, true_len], apprx) + work_time_list.max() - (work_time_list - io_time_list).max()


def worker_ps(comm, sampler, K, V, MH_max, max_len, suffix, dir):
    iters = 0
    rec = np.zeros(V, dtype=bool)
    g_theta_file = h5py.File(dir + 'tmp' + suffix + '/' + 'g_theta_file' + suffix + '.h5', 'w')
    g_theta = g_theta_file.create_dataset('g_theta', (K, V), dtype='float32')
    part_list = mk_plist(max_len, V)

    for start, end in part_list:
        dummy = bcast_np(comm, NFLT, True, root=0)
        sampler.theta[start:end, :] = dummy
        g_theta[start:end, :] = dummy

    comm.Bcast([sampler.norm_const, MPI.FLOAT], root=0)

    # wait for initial perplexity
    comm.barrier()
    w_timer = Timer(go=True)
    while not comm.Iprobe(source=0, tag=101):
        comm.isend(iters, dest=0, tag=111)

        sampler.update(MH_max, LWsampler=True, g_theta=g_theta, rec=rec)

        if comm.Iprobe(source=0, tag=102):
            comm.recv(source=0, tag=102)

            comm.Gather(np.float32(w_timer()), None, root=0)
            comm.Gather(np.float32(sampler.time_bak), None, root=0)
            g_update(comm, sampler.theta, g_theta, sampler.norm_const, K, rec, max_len)
            sampler.time_bak = 0; w_timer.go(); rec.fill(0)

        iters += 1

    comm.Gather(np.float32(w_timer()), None, root=0)
    comm.Gather(np.float32(sampler.time_bak), None, root=0)
    g_update(comm, sampler.theta, g_theta, sampler.norm_const, K, rec, max_len, send_only=True)


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


def bcast_np(comm, type, back, **kwargs):
    """
    input: either buf or xy
    """
    if 'xy' in kwargs:
        xy = kwargs.pop('xy')
        tmp = np.zeros((xy[0], xy[1]), dtype=type)
    elif 'buf' in kwargs:
        tmp = kwargs.pop('buf')
    else:
        raise ValueError('please give me either xy or buf')

    comm.Bcast([tmp, n2m(type)], **kwargs)

    if back:
        return tmp


def reduce_np(comm, type, send, **kwargs):
    """
    send is True, recvbuf will not be constructed and not return
    """
    if 'xy' in kwargs:
        xy = kwargs.pop('xy')
        tmp = np.zeros((xy[0], xy[1]), dtype=type)
    elif 'buf' in kwargs:
        tmp = kwargs.pop('buf')
    else:
        raise ValueError('please give me either xy or buf')

    if send: sendbuf = tmp; tmp = None
    else: sendbuf = np.zeros(tmp.shape, dtype=type)

    comm.Reduce([sendbuf, n2m(type)], [tmp, n2m(type)], **kwargs)

    if send: return tmp


def gather_np(comm, type, send, **kwargs):
    if 'xy' in kwargs:
        xy = kwargs.pop('xy')
        tmp = np.zeros((xy[0], xy[1]), dtype=type)
    elif 'buf' in kwargs:
        tmp = kwargs.pop('buf')
    else:
        raise ValueError('please give me either xy or buf')

    if send: sendbuf = tmp; tmp = None
    else: sendbuf = np.zeros(tmp.shape[1], dtype=type)

    comm.Gather([sendbuf, n2m(type)], [tmp, n2m(type)], **kwargs)

    # assume root=0
    if not send: return tmp[1:, :]


def est_time(disk, shape, apprx):

    return 1.5 * (1 + (disk==True)) * apprx * shape[0] * shape[1] / 1e9


def mk_mask(true_len, g_rec, rec):
    small = np.zeros(true_len, dtype=bool)
    mask_cnt = 0
    for i_g in xrange(g_rec.shape[0]):
        if g_rec[i_g]:
            if rec[i_g]: small[mask_cnt] = True
            mask_cnt += 1
    return small

if __name__ == '__main__':

    # lw_frame(5, './', '../corpus/b4_ff/', 100, int(1e5))
    lw_frame(20, '/home/lijm/WORK/yuan/', '/home/lijm/WORK/yuan/b4_ff/', 100, int(1e5), apprx=1)

    # cProfile.runctx("lw_frame(50, '../corpus/b4_ff/', 1000, int(1e5), 2)", globals(), locals(), "Profile.prof")
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()