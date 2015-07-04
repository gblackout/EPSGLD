from mpi4py import MPI
from sampler_g import *
from set_util import load_list, save_data
from sys import stdout, argv
import time
import numpy as np
import h5py
from gc import collect
# tag 110: worker send todo_list
# tag 111: worker send index of seg
# tag 112: worker send seg
# tag 100: master send index of seg
# tag 101: master send seg

def lightLDA(num, V, K, max_len, out_dir, dir, doc_per_set=int(1e4), alpha=0.01, beta=0.0001, MH_max=10):
    """ num is the num_of_samples
        dir: indicates the root folder of each data folder, tmp file folder shall be created in here"""
    fff = stdout.flush
    # ************************************ init params *******************************************************
    # init the global parameters and objects
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    seg_list = [[i*V/(size-1), (i+1)*V/(size-1)] for i in xrange(size-1)]
    bak_time = 0
    start_time = time.time()
    sampler = Gibbs_sampler(V, K, rank, doc_per_set, dir, alpha=alpha, beta=beta, word_partition=max_len,
                            single=False)
    output_name = out_dir + 'LightLDA_perplexity' + sampler.suffix + '.txt'
    if rank == 0: print 'init cnts & Bcast nkw ...'
    ndk = np.zeros((doc_per_set, K), dtype=np.int32)
    nd = np.zeros(doc_per_set, dtype=np.int32)
    z = np.array([None for _ in xrange(doc_per_set)], dtype=object)

    start = 0
    while start < V:
        end = start + max_len
        end = end * (end <= V) + V * (end > V)
        nkw_part = np.zeros((K, end-start), dtype=np.int32)
        if rank == 0:
            comm.Reduce([nkw_part.copy(), MPI.INT], [nkw_part, MPI.INT], op=MPI.SUM, root=0)
            cul_time = time.time()
            sampler.nkw[:, start:end] = nkw_part
            bak_time += time.time() - cul_time - 15 * nkw_part.shape[0] * nkw_part.shape[1] / 1e9
        else:
            sampler.init_cnts(nkw_part, ndk, nd, z, start, end, False)
            comm.Reduce([nkw_part, MPI.INT], [nkw_part.copy(), MPI.INT], op=MPI.SUM, root=0)
        start = end

    # get global mask & nkw
    sampler.mask = np.int32(sampler.mask)
    comm.Reduce([sampler.mask.copy(), MPI.INT], [sampler.mask, MPI.INT], op=MPI.SUM, root=0)
    comm.Bcast([sampler.mask, MPI.INT], root=0)
    sampler.mask = sampler.mask > 0

    comm.Reduce([sampler.nk.copy(), MPI.INT], [sampler.nk, MPI.INT], op=MPI.SUM, root=0)
    comm.Bcast([sampler.nk, MPI.INT], root=0)
    # ************************************ sampling *******************************************************
    if rank != 0:
        for _ in xrange(num):
            comm.barrier()
            todo_list = np.ones((1, len(seg_list)), dtype=bool)
            while todo_list.sum() > 0:
                comm.Send([np.int32(todo_list), MPI.INT], dest=0, tag=110)
                index = comm.recv(source=0, tag=100)
                todo_list[0, index] = 0
                nkw_part = np.zeros((K, seg_list[index][1] - seg_list[index][0]), dtype=np.int32)
                comm.Recv([nkw_part, MPI.INT], source=0, tag=101)

                sampler.update(part=seg_list[index], MH_max=MH_max, nkw_part=nkw_part, silent=True)

                comm.send(index, dest=0, tag=111)
                comm.Send([nkw_part, MPI.INT], dest=0, tag=112)

    else:
        st = MPI.Status()
        for i_n in xrange(num):
            print '*************************************************'
            print '********************* iter %i ********************' % i_n
            print '*************************************************'
            start_time = get_per_light(output_name, sampler, start_time, bak_time); bak_time = 0
            comm.barrier()

            recv_mat = np.ones((size-1, len(seg_list)), dtype=bool)
            send_mat = None
            seg_aval = np.ones((len(seg_list)), dtype=bool)
            while (recv_mat > 0).any():
                state = False

                # test if has request?
                if comm.Iprobe(source=MPI.ANY_SOURCE, tag=110):
                    state = True
                    todo_list = np.ones((1, len(seg_list)), dtype=np.int32)
                    comm.Recv([todo_list, MPI.INT], source=MPI.ANY_SOURCE, tag=110, status=st)
                    tmp = todo_list.sum()
                    todo_list = np.concatenate((todo_list, np.ones((1, 2), dtype=np.int32)*st.Get_source()), axis=1)
                    todo_list[:, -2] = tmp
                    print '---------------------------> request from: ', st.Get_source()
                    fff()
                    if send_mat is None: send_mat = todo_list
                    else: send_mat = np.concatenate((send_mat, todo_list), axis=0)

                # test if has push
                if comm.Iprobe(source=MPI.ANY_SOURCE, tag=111):
                    state = True
                    index = comm.recv(source=MPI.ANY_SOURCE, tag=111, status=st)
                    print '---------------------------> seg %i from %i' % (index, st.Get_source())
                    fff()
                    recv_mat[st.Get_source()-1, index] = 0
                    part = seg_list[index]
                    nkw_part = np.zeros((K, part[1] - part[0]), dtype=np.int32)
                    comm.Recv([nkw_part, MPI.INT], source=st.Get_source(), tag=112)
                    cul_time = time.time()
                    sampler.nkw[:, part[0]:part[1]] = nkw_part
                    bak_time += time.time() - cul_time - 15 * nkw_part.shape[0] * nkw_part.shape[1] / 1e9
                    nkw_part = None; collect()
                    seg_aval[index] = 1

                # give seg to child
                if state and seg_aval.sum() > 0 and send_mat is not None:
                    send_mat = send_mat[send_mat[:, -2].argsort()]
                    send_mat_mask = np.ones(send_mat.shape[0], dtype=bool)
                    for i_o in reversed(range(send_mat.shape[0])):
                        min = 2*size
                        min_ind = 0
                        for i in xrange(send_mat.shape[1]-2):
                            if seg_aval[i] and send_mat[i_o, i] and send_mat[:i_o+1, i].sum() < min:
                                min = send_mat[:i_o+1, i].sum()
                                min_ind = i
                        if min < 2*size:
                            dest = send_mat[i_o, -1]
                            comm.isend(min_ind, dest=dest, tag=100)
                            seg_aval[min_ind] = 0
                            send_mat_mask[i_o] = 0

                            part = seg_list[min_ind]
                            cul_time = time.time()
                            nkw_part = sampler.nkw[:, part[0]:part[1]]
                            bak_time += time.time() - cul_time - 15 * nkw_part.shape[0] * nkw_part.shape[1] / 1e9
                            print '---------------------------> send seg %i to %i' % (min_ind, dest)
                            fff()
                            comm.Send([nkw_part, MPI.INT], dest=dest, tag=101)
                            nkw_part = None; collect()
                    if send_mat_mask.sum() > 0: send_mat = send_mat[send_mat_mask, :]
                    else: send_mat = None

        get_per_light(output_name, sampler, start_time, bak_time)


def get_per_light(output_name, sampler, start_time, bak_time):
    f = open(output_name, 'a')
    start_time += bak_time
    per_s = time.time()
    print '---------------------------> computing perplexity: '
    prplx = sampler.get_perp_just_in_time(50, 10)
    print '---------------------------> perplexity: %.2f' % prplx
    f.write('%.2f\t%.2f\n' % (prplx, per_s - start_time))
    f.close()
    return start_time + time.time() - per_s


if __name__ == '__main__':
    # # small_set
    # lightLDA(30, 2, V=5, K=5, dir='./small_test_light_d/', doc_per_set=15)

    # very large
    lightLDA(25, int(1e5), 1000, 10000, '/home/lijm/WORK/yuan/', '/home/lijm/WORK/yuan/t_saved/')

