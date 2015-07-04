
import numpy as np
import processwiki as pw
import util_funcs
from sys import argv
import time
from os import listdir, remove, makedirs
import tables as tb
import re
import pstats, cProfile
import warnings
from scipy import sparse
from scipy.io import mmread, mmwrite
import multiprocessing as mp
import multiprocessing.sharedctypes
import ctypes

class Gibbs_sampler(object):

    # run: 8G; out: 0G
    def __init__(self, V, K, rank, doc_per_set, dir='./', alpha=0.01, beta=0.0001, word_partition=2000, set_name='t_saved',
                 test_doc='test_doc', ps=2):
        """ dir: indicates the root folder of each data folder, tmp file folder shall be created in here
            NOTICE: mask is incomplete if dis, manually add it
            NOTICE: digits in set name are used to ident the ndk nd and z"""
        # ******************************* store parameter *********************************************
        self.K = K
        self.V = V
        self.doc_per_set = doc_per_set
        self.suffix = time.strftime('_%m%d_%H%M%S', time.localtime()) + str(rank)
        self.data_dir = dir + str(rank) + '/'
        self.tmp_dir = dir + 'tmp' + self.suffix + '/'
        makedirs(self.tmp_dir)
        self.alpha = alpha
        self.beta = beta
        self.beta_bar = beta * V
        self.alpha_bar = alpha * K
        self.mask = np.ones(V, dtype=bool)
        util_funcs.set_srand()
        self.set_name = set_name
        self.ps = ps
        self.rank = rank

        # ******************************* init the matrices *********************************************
        self.name = self.tmp_dir + 'nkw' + self.suffix + '.h5'
        self.node_name = 'nkw'
        nkw_file = tb.open_file(self.name, mode='w', title=self.node_name)
        nkw_file.create_array(nkw_file.root, self.node_name, atom=tb.Int32Atom(), shape=(K, self.V))
        nkw_file.close()
        self.nk = np.zeros(K, dtype=np.int32)

        self.mask = np.zeros(self.V, dtype=bool)
        self.test_doc = np.load(self.data_dir + test_doc + '.npy').tolist()

        # ******************************* init counts&mask *********************************************
        ndk = np.zeros((self.doc_per_set, self.K), dtype=np.int32)
        nd = np.zeros(self.doc_per_set, dtype=np.int32)
        z = np.array([None for _ in xrange(self.doc_per_set)], dtype=object)

        start = 0
        while start < self.V:
            end = start + word_partition
            end = end * (end <= self.V) + self.V * (end > self.V)

            nkw_part, raw_nkw_part = load_table((self.K, end - start), ctypes.c_int32, self.name, self.node_name,
                                                [i for i in xrange(start, end)], self.ps, type=np.int32)
            for file_name in listdir(self.data_dir):
                if self.set_name in file_name:
                    print '(%i)%i: %i init set: ' % (rank, start, end), file_name
                    train_cts = np.load(self.data_dir + file_name).tolist()
                    if start != 0: (ndk, nd, z) = self.load_d(re.search(r'\d+', file_name).group())
                    util_funcs.init_light(train_cts, ndk, nkw_part, nd, self.nk, self.doc_per_set, z,
                                          self.K, start, end, start == 0)
                    self.save_and_init(re.search(r'\d+', file_name).group(), ndk, nd, z)
                    ndk.fill(0)
                    nd.fill(0)
                    z = np.array([None for _ in xrange(self.doc_per_set)], dtype=object)

            self.mask[start:end] = ~(nkw_part == 0).all(0)
            write_table(raw_nkw_part, nkw_part, nkw_part.shape, self.name, self.node_name,
                        [i for i in xrange(start, end)], self.ps, type=np.int32)

            start = end
        # debug
        # if rank != 0:
        #     file = tb.open_file(self.name, 'r+')
        #     nkw = file.get_node(file.root, self.node_name)[:, :]
        #     print rank, (nkw[:, self.mask].sum(0) > 0).any()
        #     print rank, (nkw[:, self.mask].sum(0) <= 1e-8).any()
        #     print rank, ((file.get_node(file.root, self.node_name)[:, :].sum(1) - self.nk) != 0).any()


    def load_d(self, set_name):
        ndk = mmread(self.tmp_dir + self.suffix + set_name + '_ndk').toarray()
        nd = np.load(self.tmp_dir + self.suffix + set_name + '_nd')
        z = np.load(self.tmp_dir + self.suffix + set_name + '_zzz' + '.npy')

        return ndk, nd, z

    def save_and_init(self, set_name, ndk, nd, z):
        mmwrite(self.tmp_dir + self.suffix + set_name + '_ndk', sparse.lil_matrix(ndk))
        nd.dump(self.tmp_dir + self.suffix + set_name + '_nd')
        np.save(self.tmp_dir + self.suffix + set_name + '_zzz', z)

    # run: 24G; out: 0G
    def update(self, part, MH_max=None, nkw_part=None):
        """ leave MH_max blank if gibbs is True, otherwise you must specify it"""

        # file = tb.open_file(self.name, 'r+')
        # nkw = file.get_node(file.root, self.node_name)[:, :]
        # print self.rank, (nkw[:, self.mask].sum(0) > 0).any()
        # print self.rank, (nkw[:, self.mask].sum(0) <= 1e-8).any()
        # print self.rank, ((file.get_node(file.root, self.node_name)[:, :].sum(1) - self.nk) != 0).any()
        # file.close()


        if nkw_part is None:
            nkw_part, raw_nkw_part = load_table((self.K, part[1] - part[0]), ctypes.c_int32, self.name, self.node_name,
                                                [i for i in xrange(part[0], part[1])], self.ps, type=np.int32)

        for file_name in listdir(self.data_dir):
            if self.set_name in file_name:
                print '(%i)%i: %i set: ' % (self.rank, part[0], part[1]), file_name
                (table_h, table_l, table_p) = self.gen_table(part, nkw_part)
                nkw_stale = nkw_part.copy()
                nk_stale = np.copy(self.nk)
                train_cts = np.load(self.data_dir + file_name).tolist()
                (ndk, nd, z) = self.load_d(re.search(r'\d+', file_name).group())

                util_funcs.sample_light(self.K, self.alpha, self.alpha_bar, self.beta, self.beta_bar, table_h, table_l,
                                        table_p, nkw_part, self.nk, ndk, nd, nkw_stale, nk_stale, z,
                                        train_cts, MH_max, part[0], part[1], self.doc_per_set, file_name)

                self.save_and_init(re.search(r'\d+', file_name).group(), ndk, nd, z)

        if nkw_part is None:
            write_table(raw_nkw_part, nkw_part, nkw_part.shape, self.name, self.node_name,
                        [i for i in xrange(part[0], part[1])], self.ps, type=np.int32)

    # run: 16G; out: 12G
    def gen_table(self, part, nkw_part):

        # we do not use mask here cause the gen_table will handle it
        phi = np.float32(nkw_part) / np.float32(self.nk[:, np.newaxis])
        phi[:, self.mask[part[0]:part[1]]] /= np.sum(phi[:, self.mask[part[0]:part[1]]], 0)
        # note table is V by K
        table_h = np.zeros((phi.shape[1], phi.shape[0]), dtype=np.int32)
        table_l = np.zeros((phi.shape[1], phi.shape[0]), dtype=np.int32)
        table_p = np.zeros((phi.shape[1], phi.shape[0]), dtype=np.float32)

        util_funcs.gen_alias_table_light(table_h, table_l, table_p, phi, self.mask[part[0]:part[1]])

        return table_h, table_l, table_p

    # run: 8G; out: 0G
    def get_perp_just_in_time(self, iter, MH_max):
        # *************************************** parameters ************************************************
        phi_mask = np.logical_and(self.test_doc[2], self.mask)
        nkw_part, raw_nkw_part = load_table((self.K, phi_mask.sum()), ctypes.c_int32, self.name, self.node_name,
                                            self.to_list(phi_mask), self.ps, type=np.int32)
        phi = np.float32(nkw_part) / np.float32(self.nk[:, np.newaxis])

        nkw_part = None
        raw_nkw_part = None
        samples = util_funcs.gen_obj(phi.shape[1])
        table_h = np.zeros(self.K, dtype=np.int32)
        table_l = np.zeros(self.K, dtype=np.int32)
        table_p = np.zeros(self.K, dtype=np.float32)
        batch_map = np.zeros(self.V, dtype=np.int32)
        util_funcs.gen_batch_map(phi_mask, batch_map, self.V)

        # *************************************** sampling ************************************************
        util_funcs.gen_alias_table(table_h=table_h, table_l=table_l, table_p=table_p, phi=phi/np.sum(phi, 0),
                                   batch_mask=phi_mask, w_sample=self.test_doc[3],
                                   samples=samples, iter_per_update=iter, MH_max=MH_max)

        batch_N = sum(len(doc) for doc in self.test_doc[0])
        batch_D = len(self.test_doc[0])
        w_cnt = phi.shape[1]
        z = [None for _ in xrange(batch_D)]
        Adk = np.zeros((batch_D, self.K), dtype=np.int32)
        Adk_mean = np.zeros(Adk.shape, dtype=np.float32)
        burn_in = iter // 2
        rand_kkk = np.random.randint(self.K, size=batch_N)

        util_funcs.sample_z_par_alias_per(batch_D, self.test_doc[0], z, w_cnt, self.K, iter, burn_in, self.alpha,
                                          self.alpha_bar, self.beta, self.beta_bar, Adk, Adk_mean, batch_map, phi,
                                          samples, MH_max, rand_kkk, phi_mask, True)
        # *************************************** perplexity ************************************************
        Adk_mean += self.alpha
        Adk_mean /= np.sum(Adk_mean, 1)[:, np.newaxis]

        doc_len = len(self.test_doc[1])
        log_avg_probs = 0

        for d in xrange(doc_len):
            for w in self.test_doc[1][d]:
                if not self.mask[w]:
                    continue
                log_avg_probs += np.log(np.dot(Adk_mean[d, :], phi[:, batch_map[w]]))

        num = sum([len(d) for d in self.test_doc[1]])
        util_funcs.kill_obj(phi.shape[1], samples)
        return np.exp(- log_avg_probs / num)

    def to_list(self, mask):
        mask_list = []
        for i in xrange(mask.shape[0]):
            if mask[i]: mask_list += [i]
        return mask_list

    @staticmethod
    def clear_tmp(dir):
        for file_name in listdir(dir):
            if ('ndk' in file_name) or ('nd' in file_name) or ('zzz' in file_name):
                remove(dir+file_name)


def slice_list(input, size):
    input_size = len(input)
    slice_size = input_size / size
    remain = input_size % size
    result = []
    iterator = iter(input)
    for i in range(size):
        result.append([])
        for j in range(int(slice_size)):
            result[i].append(iterator.next())
        if remain:
            result[i].append(iterator.next())
            remain -= 1
    return result


def load_table(shape, c_type, name, node_name, selection, ps, type=np.float32, axis=1, rank=1):
    """ multi-ps load pytable into sharedmem
        shape: shape of numpy arr
        c_type: types from ctypes module
        name: pytable file
        node_name: pytable node name
        selection: list of indices of pytable to be loaded
        ps: num of ps to be used
        type: for nkw please use np.int32
        axis: along which the selection is made
        return: theta, raw_theta"""
    if ps == 1:
        file = tb.open_file(name, mode='r+')
        if axis: theta = file.get_node(file.root, node_name)[:, selection]
        else: theta = file.get_node(file.root, node_name)[selection, :]
        file.close()
        return theta, None
    else:
        raw_theta = mp.sharedctypes.RawArray(c_type, shape[0] * shape[1])
        ps_rec = []
        sel_list = slice_list(selection, ps)
        offset = 0
        for i in xrange(ps):
            # debug
            # if rank == 0: print 'load-->', i, sel_list[i][0], sel_list[i][-1], offset, len(sel_list[i])
            process = mp.Process(target=load_table_ps,
                                 args=(raw_theta, shape, name, node_name, sel_list[i], offset, type, axis))
            offset += len(sel_list[i])
            process.start()
            ps_rec.append(process)

        for e in ps_rec:
            e.join()

        t = to_np(raw_theta, type)
        t.shape = shape
        return t, raw_theta


def load_table_ps(raw_theta, shape, name, node_name, selection, offset, type, axis):
    if len(selection) == 0:
        return
    theta = to_np(raw_theta, type)
    theta.shape = shape
    file = tb.open_file(name, 'r+')
    if axis: theta[:, offset:offset+len(selection)] = file.get_node(file.root, node_name)[:, selection]
    else: theta[offset:offset+len(selection), :] = file.get_node(file.root, node_name)[selection, :]
    file.close()


def write_table(raw_theta, theta, shape, name, node_name, selection, ps, type=np.float32, axis=1, rank=1):
    """ multi-ps write to pytable
        raw_theta, theta: former used if ps > 1, you should leave either one None according to ps
        shape: shape of numpy arr
        name: pytable file
        node_name: pytable node name
        selection: list of indices of pytable to be loaded
        ps: num of ps to be used
        type: for nkw please use np.int32"""
    if ps == 1:
        file = tb.open_file(name, 'a')
        if axis: file.get_node(file.root, node_name)[:, selection] = theta
        else: file.get_node(file.root, node_name)[selection, :] = theta
        file.close()
    else:
        ps_rec = []
        sel_list = slice_list(selection, ps)
        offset = 0
        for i in xrange(ps):
            # if rank == 0 and len(sel_list[i]) != 0:
            #     print 'write-->', i, min(sel_list[i]), max(sel_list[i]), offset, len(sel_list[i])
            process = mp.Process(target=write_table_ps,
                                 args=(raw_theta, shape, name, node_name, sel_list[i], offset, type, axis, rank))
            offset += len(sel_list[i])
            process.start()
            ps_rec.append(process)
            # if rank == 0:
            #     time.sleep(1)
            #     file = tb.open_file(name, 'r+')
            #     print 'inside load medium', file.get_node(file.root, node_name)[-1, :10], offset
            #     file.close()

        for e in ps_rec:
            e.join()


def write_table_ps(raw_theta, shape, name, node_name, selection, offset, type, axis, rank=1):
    if len(selection) == 0:
        return
    theta = to_np(raw_theta, type)
    theta.shape = shape
    file = tb.open_file(name, 'a')
    if axis:
        # if rank == 0:
        #     print 'inside load before', file.get_node(file.root, node_name)[-1, :10], offset
        file.get_node(file.root, node_name)[:, selection] = theta[:, offset: offset+len(selection)]
        # if rank == 0:
        #     print 'inside load ', file.get_node(file.root, node_name)[-1, :10], offset
    else: file.get_node(file.root, node_name)[selection, :] = theta[offset: offset+len(selection), :]
    file.close()


def to_np(raw_arr, type):
    return np.frombuffer(raw_arr, dtype=type)



def run_very_large_light(MH_max):
    num = 30
    rank = 0
    word_partition = int(1e4)
    doc_per_set = int(1e4)
    V = int(1e5)
    K = 100
    jump = 5
    jump_bias = 10
    jump_hold = 0
    # dir = './light/%i/' % offset
    dir = '../corpus/t_saved/'

    # num = 100
    # offset = 0
    # word_partition = 2
    # doc_per_set = 15
    # V = 5
    # K = 5
    # jump = 10
    # jump_bias = 10
    # jump_hold = 0
    # set_name = 'saved'
    # dir = './small_test_light/'

    output_name = 'serial_light_perplexity' + time.strftime('_%m%d_%H%M%S', time.localtime()) + '.txt'

    start = time.time()
    sampler = Gibbs_sampler(V, K, rank, doc_per_set, dir, word_partition=word_partition, )

    f = open(output_name, 'w')

    print 'computing perplexity: ', -1
    f.write('%.2f\t%.2f\n' % (sampler.get_perp_just_in_time(50, 10), 0))

    for i in xrange(num):
        print 'iter--->', i

        start_w = 0
        while start_w < V:
            end = start_w + word_partition
            end = end * (end <= V) + V * (end > V)

            sampler.update([start_w, end], MH_max=MH_max)

            start_w = end

        if i < jump_bias:
            per_s = time.time()
            print 'computing perplexity: ', i
            f.write('%.2f\t%.2f\n' % (sampler.get_perp_just_in_time(50, 10), per_s - start))
            start += time.time() - per_s
        elif (i + 1) % jump == 0 and (i + 1) >= jump_hold:
            per_s = time.time()
            print 'computing perplexity: ', i
            f.write('%.2f\t%.2f\n' % (sampler.get_perp_just_in_time(50, 10), per_s - start))
            start += time.time() - per_s

    f.close()


if __name__ == '__main__':
    run_very_large_light(10)

    # cProfile.runctx("run_very_large_light(10)", globals(), locals(), "Profile.prof")
    #
    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()
