from mpi4py import MPI
from sampler_g import *
from set_util import load_list, save_data
from sys import stdout, argv
import time
import numpy as np
import multiprocessing as mp
import multiprocessing.sharedctypes as ms
import ctypes
import tables as tb

# tag 110: worker send todo_list
# tag 111: worker send index of seg
# tag 112: worker send seg
# tag 100: master send index of seg
# tag 101: master send seg

def lightLDA(num, ps, V=int(1e5), K=1000, max_len=15000, dir='./', doc_per_set=1e4, alpha=0.01, beta=0.0001, MH_max=10):
    """ num is the num_of_samples
        dir: indicates the root folder of each data folder, tmp file folder shall be created in here"""
    fff = stdout.flush
    # ************************************ init params *******************************************************
    # init the global parameters and objects
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    seg_list = [[i*V/(size-1), (i+1)*V/(size-1)] for i in xrange(size-1)]

    sampler = Gibbs_sampler(V, K, rank, doc_per_set, dir, alpha=alpha, beta=beta,
                            word_partition=max_len, ps=ps)

    # get global mask & nkw
    sampler.mask = np.int32(sampler.mask)
    comm.Reduce([sampler.mask.copy(), MPI.INT], [sampler.mask, MPI.INT], op=MPI.SUM, root=0)
    comm.Bcast([sampler.mask, MPI.INT], root=0)
    sampler.mask = sampler.mask > 0

    comm.Reduce([sampler.nk.copy(), MPI.INT], [sampler.nk, MPI.INT], op=MPI.SUM, root=0)
    comm.Bcast([sampler.nk, MPI.INT], root=0)

    if rank == 0: print 'Bcast nkw ...'
    start = 0
    while start < V:
        end = start + max_len
        end = end * (end <= V) + V * (end > V)
        nkw_part, raw_nkw_part = load_table((K, end - start), ctypes.c_int32, sampler.name, sampler.node_name,
                                            [i for i in xrange(start, end)], ps, type=np.int32, rank=rank)

        if rank == 0: comm.Reduce([nkw_part.copy(), MPI.INT], [nkw_part, MPI.INT], op=MPI.SUM, root=0)
        else: comm.Reduce([nkw_part, MPI.INT], [nkw_part.copy(), MPI.INT], op=MPI.SUM, root=0)

        comm.Bcast([nkw_part, MPI.INT], root=0)
        write_table(raw_nkw_part, nkw_part, nkw_part.shape, sampler.name, sampler.node_name,
                    [i for i in xrange(start, end)], ps, type=np.int32, rank=rank)

        start = end


    # file = tb.open_file(sampler.name, 'r+')
    # nkw = file.get_node(file.root, sampler.node_name)[:, :]
    # file.close()
    # if rank == 0:
    #     nkw[:, :] = 0
    #     comm.Reduce([nkw.copy(), MPI.INT], [nkw, MPI.INT], op=MPI.SUM, root=0)
    # else:
    #     comm.Reduce([nkw, MPI.INT], [nkw.copy(), MPI.INT], op=MPI.SUM, root=0)
    #
    # nkw_1 = None
    # nkw_2 = None
    # if rank != 0:
    #     comm.Send([nkw, MPI.INT], dest=0)
    # else:
    #     nkw_1 = nkw.copy()
    #     nkw_2 = nkw.copy()
    #     comm.Recv([nkw_1, MPI.INT], source=1)
    #     comm.Recv([nkw_2, MPI.INT], source=2)
    #
    # if rank == 0:
    #     file = tb.open_file(sampler.name, 'r+')
    #     nkw_ps = file.get_node(file.root, sampler.node_name)[:, :]
    #     file.close()
    #     flag = (nkw.sum(0) - nkw_ps.sum(0)) > 0
    #     print '************', flag.any()
    #     if flag.any():
    #         for i in xrange(flag.shape[0]):
    #             if flag[i]:
    #                 print '--------------------'
    #                 print 'a', i, nkw[:, i], nkw[:, i].sum()
    #                 print 'b', i, nkw_ps[:, i], nkw_ps[:, i].sum()
    #                 print '1', i, nkw_1[:, i], nkw_1[:, i].sum()
    #                 print '2', i, nkw_2[:, i], nkw_2[:, i].sum()

    # ************************************ sampling *******************************************************
    if rank != 0:
        for _ in xrange(num):
            todo_list = np.ones((1, len(seg_list)), dtype=bool)
            while todo_list.sum() > 0:
                comm.Send([np.int32(todo_list), MPI.INT], dest=0, tag=110)
                index = comm.recv(source=0, tag=100)
                todo_list[0, index] = 0
                nkw_part = np.zeros((K, seg_list[index][1] - seg_list[index][0]), dtype=np.int32)
                comm.Recv([nkw_part, MPI.INT], source=0, tag=101)

                sampler.update(part=seg_list[index], MH_max=MH_max, nkw_part=nkw_part)

                comm.send(index, dest=0, tag=111)
                comm.Send([nkw_part, MPI.INT], dest=0, tag=112)

    else:
        st = MPI.Status()
        start = time.time()
        f = open('LightLDA_perplexity' + sampler.suffix +'.txt', 'w')
        for i_n in xrange(num):
            print '*************************************************'
            print '********************* iter %i ********************' % i_n
            print '*************************************************'
            fff()
            get_per(f, sampler, start)

            todo_mat = np.ones((size-1, len(seg_list)), dtype=bool)
            send_mat = None
            seg_aval = np.ones((len(seg_list)), dtype=bool)
            while (todo_mat > 0).any():
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
                    part = seg_list[index]
                    raw_nkw_part = ms.RawArray(ctypes.c_float, K * (part[1] - part[0]))
                    nkw_part = to_np(raw_nkw_part, np.int32)
                    nkw_part.shape = (K, part[1] - part[0])
                    comm.Recv([nkw_part, MPI.INT], source=st.Get_source(), tag=112)
                    write_table(raw_nkw_part, nkw_part, nkw_part.shape, sampler.name, sampler.node_name,
                                [i for i in xrange(part[0], part[1])], ps, type=np.int32)
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
                            todo_mat[dest-1, min_ind] = 0
                            send_mat_mask[i_o] = 0

                            part = seg_list[min_ind]
                            nkw_part, raw_nkw_part = load_table((K, part[1] - part[0]), ctypes.c_int32, sampler.name,
                                                                sampler.node_name, [i for i in xrange(part[0], part[1])],
                                                                ps, type=np.int32)
                            print '---------------------------> send seg %i to %i' % (min_ind, dest)
                            fff()
                            comm.Send([nkw_part, MPI.INT], dest=dest, tag=101)
                    if send_mat_mask.sum() > 0: send_mat = send_mat[send_mat_mask, :]
                    else: send_mat = None

        get_per(f, sampler, start)
        f.close()

def get_per(f, sampler, start_time):
    per_s = time.time()
    print '---------------------------> computing perplexity: '
    f.write('%.2f\t%.2f\n' % (sampler.get_perp_just_in_time(50, 10), per_s - start_time))
    return start_time + time.time() - per_s

if __name__ == '__main__':
    # # small_set
    # lightLDA(30, 2, V=5, K=5, dir='./small_test_light_d/', doc_per_set=15)

    # very large
    lightLDA(5, 2, V=int(1e5), K=100, dir='../corpus/t_saved/', doc_per_set=int(1e4))

