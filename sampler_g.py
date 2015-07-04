import numpy as np
import processwiki as pw
import util_funcs
from sys import argv
import time
from os import listdir, remove, makedirs
import h5py
import re
import pstats, cProfile
import warnings
from scipy import sparse
import threading


class Loader(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        self.arr = None
        self.path = path

    def run(self):
        self.arr = np.load(self.path)


class Dumper(threading.Thread):
    def __init__(self, path, arr):
        threading.Thread.__init__(self)
        self.arr = arr
        self.path = path

    def run(self):
        self.arr.dump(self.path)


class Gibbs_sampler(object):
    # run: 8G; out: 0G
    def __init__(self, V, K, rank, doc_per_set, dir='./', alpha=0.01, beta=0.0001, word_partition=2000, set_name='t_saved',
                 test_doc='test_doc', silent=False, single=True):
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
        self.rank = rank
        self.bak_time = 0
        # ******************************* init the matrices *********************************************
        self.name = self.tmp_dir + 'nkw' + self.suffix + '.h5'
        self.node_name = 'nkw'
        nkw_file = h5py.File(self.tmp_dir + 'nkw' + self.suffix + '.h5', 'w')
        self.nkw = nkw_file.create_dataset('nkw', (K, self.V), dtype='int32')
        self.nk = np.zeros(K, dtype=np.int32)

        self.mask = np.zeros(self.V, dtype=bool)
        self.test_doc = np.load(self.data_dir + test_doc + '.npy').tolist()

        # ******************************* init counts&mask *********************************************
        self.train_cts_set = []
        for file_name in listdir(self.data_dir):
            if self.set_name in file_name:
                self.train_cts_set.append((file_name, np.load(self.data_dir + file_name).tolist()))

        if single:
            ndk = np.zeros((self.doc_per_set, self.K), dtype=np.int32)
            nd = np.zeros(self.doc_per_set, dtype=np.int32)
            z = np.array([None for _ in xrange(self.doc_per_set)], dtype=object)

            start = 0
            while start < self.V:
                end = start + word_partition
                end = end * (end <= self.V) + self.V * (end > self.V)

                nkw_part = np.zeros((K, end - start), dtype=np.int32)
                self.init_cnts(nkw_part, ndk, nd, z, start, end, silent)
                self.nkw[:, start:end] = nkw_part

                start = end

    def init_cnts(self, nkw_part, ndk, nd, z, start, end, silent):
        """ used for init_cnts, the load and write should be handled outside
            the init of ndk, nk, z should also be handled outside"""
        for file_name, train_cts in self.train_cts_set:
            if not silent: print '(%i)%i: %i init set: ' % (self.rank, start, end), file_name

            if start != 0: (ndk, nd, z) = self.load_d(re.search(r'\d+', file_name).group())

            util_funcs.init_light(train_cts, ndk, nkw_part, nd, self.nk, self.doc_per_set, z,
                                  self.K, start, end, start == 0)

            self.save_and_init(re.search(r'\d+', file_name).group(), ndk, nd, z)
            ndk.fill(0); nd.fill(0)
            z = np.array([None for _ in xrange(self.doc_per_set)], dtype=object)

        self.mask[start:end] = ~(nkw_part == 0).all(0)

    def load_d(self, set_name):
        pn = self.tmp_dir + self.suffix + set_name

        ndk = sparse.csr_matrix((np.load(pn + '_ndk_data'), np.load(pn + '_ndk_indices'), np.load(pn + '_ndk_indptr')),
                                shape=(self.doc_per_set, self.K)).toarray()
        nd = np.load(pn + '_nd')
        z = np.load(pn + '_zzz' + '.npy')

        # name_list = [pn + '_ndk_data', pn + '_ndk_indices', pn + '_ndk_indptr', pn + '_nd', pn + '_zzz' + '.npy']
        # t_rec = []
        #
        # for name in name_list:
        #     t = Loader(name)
        #     t.start()
        #     t_rec.append(t)
        #
        # for t in t_rec:
        #     t.join()
        #
        # ndk = sparse.csr_matrix((t_rec[0].arr, t_rec[1].arr, t_rec[2].arr), shape=(self.doc_per_set, self.K)).toarray()
        # nd = t_rec[3].arr
        # z = t_rec[4].arr

        return ndk, nd, z

    def save_and_init(self, set_name, ndk, nd, z):
        pn = self.tmp_dir + self.suffix + set_name

        start = time.time()
        ndk = sparse.csr_matrix(ndk)
        self.bak_time += time.time() - start

        ndk.data.dump(pn + '_ndk_data')
        ndk.indices.dump(pn + '_ndk_indices')
        ndk.indptr.dump(pn + '_ndk_indptr')
        nd.dump(pn + '_nd')
        np.save(pn + '_zzz', z)


        # t_rec = []
        # ndk = sparse.csr_matrix(ndk)
        # name_list = [(ndk.data, pn + '_ndk_data'), (ndk.indices, pn + '_ndk_indices'), (ndk.indptr, pn + '_ndk_indptr'),
        #              (nd, pn + '_nd')]
        #
        # for arr, name in name_list:
        #     t = Dumper(name, arr)
        #     t.start()
        #     t_rec.append(t)
        #
        # np.save(pn + '_zzz', z)
        #
        # for t in t_rec:
        #     t.join()

    # run: 24G; out: 0G
    def update(self, part, MH_max=None, nkw_part=None, silent=False):
        """ leave MH_max blank if gibbs is True, otherwise you must specify it"""

        if nkw_part is None:
            cul_time = time.time()
            nkw_part = self.nkw[:, part[0]:part[1]]
            self.bak_time += time.time() - cul_time - 15 * nkw_part.shape[0] * nkw_part.shape[1] / 1e9

        (table_h, table_l, table_p) = self.gen_table(part, nkw_part)

        for file_name, train_cts in self.train_cts_set:
            if not silent: print '(%i)%i: %i set: ' % (self.rank, part[0], part[1]), file_name

            nkw_stale = nkw_part.copy()
            nk_stale = np.copy(self.nk)
            (ndk, nd, z) = self.load_d(re.search(r'\d+', file_name).group())

            util_funcs.sample_light(self.K, self.alpha, self.alpha_bar, self.beta, self.beta_bar, table_h, table_l,
                                    table_p, nkw_part, self.nk, ndk, nd, nkw_stale, nk_stale, z,
                                    train_cts, MH_max, part[0], part[1], self.doc_per_set, file_name)

            self.save_and_init(re.search(r'\d+', file_name).group(), ndk, nd, z)

        if nkw_part is None:
            cul_time = time.time()
            self.nkw[:, part[0]:part[1]] = nkw_part
            self.bak_time += time.time() - cul_time - 15 * nkw_part.shape[0] * nkw_part.shape[1] / 1e9

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
        phi = np.float32(self.nkw[:, phi_mask]) / np.float32(self.nk[:, np.newaxis])

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


def run_very_large_light(MH_max):
    num = 1
    rank = 1
    word_partition = 10000
    doc_per_set = int(1e4)
    V = int(1e5)
    K = 1000
    jump = 5
    jump_bias = 10
    jump_hold = 0
    # dir = './light/%i/' % offset
    dir = '/home/lijm/WORK/yuan/t_saved/'
    out_dir = '/home/lijm/WORK/yuan/'

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

    output_name = out_dir+'serial_light_perplexity' + time.strftime('_%m%d_%H%M%S', time.localtime()) + '.txt'

    start_time = time.time()
    sampler = Gibbs_sampler(V, K, rank, doc_per_set, dir, word_partition=word_partition)

    f = open(output_name, 'w')
    # start_time = get_per(f, sampler, start_time)
    start_time = time.time()

    for i in xrange(num):
        print 'iter--->', i

        start_w = 0
        while start_w < V:
            end = start_w + word_partition
            end = end * (end <= V) + V * (end > V)

            sampler.update([start_w, end], MH_max=MH_max)

            start_w = end

        # if i < jump_bias:
        #     start_time = get_per(f, sampler, start_time)
        # elif (i + 1) % jump == 0 and (i + 1) >= jump_hold:
        #     start_time = get_per(f, sampler, start_time)
    print start_time - sampler.bak_time
    f.close()


def get_per(f, sampler, start_time):
    start_time += sampler.bak_time; sampler.bak_time = 0
    per_s = time.time()
    print '---------------------------> computing perplexity: '
    f.write('%.2f\t%.2f\n' % (sampler.get_perp_just_in_time(50, 10), per_s - start_time))
    return start_time + time.time() - per_s

if __name__ == '__main__':
    # run_very_large_light(2)

    cProfile.runctx("run_very_large_light(2)", globals(), locals(), '/home/lijm/WORK/yuan/'+"Profile.prof")

    s = pstats.Stats('/home/lijm/WORK/yuan/'+"Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
