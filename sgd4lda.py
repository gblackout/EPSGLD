from __future__ import division

import pstats, cProfile

import numpy as np
import h5py
import util_funcs
from sys import stdout
import time
from set_util import load_list
from pickle import load
from os import path, makedirs
from gc import collect

class LDSampler(object):
    """ H: value sqrt(m)*sigma^(1+0.3)
        dir: indicates the root folder of each data folder, tmp file folder shall be created in here
        rank: indicate the subfolder where the docs exist
        D: the total docs in the local training set
        K: the num of topic
        W: the len of vocab
        test_doc: indicates the name of test_doc
        time_bak: time used to load, correct with 1.5s; you need to set 0 every time you use it

        train_set: has form [ [[d],[d],[d],[d]], map[], mask[], flag, [maskd[], maskd[], maskd[], maskd[]] ]
        test_doc: [ [[w], [..], ...], [[test_w], [..], ..], mask[], map[] ]
        """
    # (max_len/10000) * 12 G
    def __init__(self, H, dir, rank, D, K, W, max_len, apprx, batch_size=50, alpha=0.01, beta=0.0001,
                 epsilon=0.01, tau0=1000, kappa=0.55, samples_per_update=50, test_doc='test_doc', suffix=None):
        # set the related parameters1
        self.K = K
        self.batch_size = batch_size
        self.step_size_params = (epsilon, tau0, kappa)
        self.samples_per_update = samples_per_update

        self.W = W
        self.D = D
        self.H = H

        self.alpha = alpha
        self.beta = beta
        self.beta_bar = beta * self.W
        self.alpha_bar = alpha * K

        self.update_ct = 0
        self.rank = rank
        if suffix is None: suffix = time.strftime('_%m%d_%H%M%S', time.localtime()) + '_' + str(rank)
        self.data_dir = dir + str(rank) + '/'
        self.tmp_dir = dir + 'tmp' + suffix + '/'
        makedirs(self.tmp_dir)
        self.current_set = None
        self.batch_loc = [0, 0]
        self.time_bak = 0
        self.apprx = apprx

        # used to map between real w and the sliced cnts matrix in memory
        self.batch_map = np.zeros(self.W, dtype=np.int32)
        self.batch_map_4w = np.zeros(self.W, dtype=np.int32)
        self.w4_cnt = None
        util_funcs.set_srand()

        # allocate the file
        theta_file = h5py.File(self.tmp_dir + 'theta' + suffix + '.h5', 'w')
        self.theta = theta_file.create_dataset('theta', (K, self.W), dtype='float32')
        self.norm_const = np.zeros((self.K, 1), dtype=np.float32)

        start = 0
        while start < self.W:
            end = start + max_len
            end = end * (end <= self.W) + self.W * (end > self.W)
            tmp = np.random.gamma(1, 1, (self.K, end-start)); collect()
            self.theta[:, start:end] = tmp
            self.norm_const[:] += np.sum(tmp, 1)[:, np.newaxis]

            start = end
            tmp = None; collect()

        self.ndk = np.zeros((self.batch_size, K), dtype=np.int32)
        self.ndk_avg = np.zeros((self.batch_size, K), dtype=np.float32)
        self.nd = np.zeros(self.batch_size, dtype=np.int32)
        self.nk = np.zeros(K, dtype=np.int32)

        self.table_h = np.zeros(self.K, dtype=np.int32)
        self.table_l = np.zeros(self.K, dtype=np.int32)
        self.table_p = np.zeros(self.K, dtype=np.float32)
        self.samples = None
        self.test_doc = load(open(self.data_dir + test_doc, 'r'))

        self.mask = np.ones(self.W, dtype=bool)

        self.iters_per_doc = 50

    # space: 20G
    def update(self, MH_max, LWsampler=False, g_theta=None, rec=None):
        train_cts, phi = self.next_batch(MH_max); collect()

        batch_mask = self.current_set[4][self.batch_loc[1]]
        if LWsampler: rec[:] = rec + batch_mask
        util_funcs.gen_batch_map(batch_mask, self.batch_map, self.W)

        cul_time = time.time()
        batch_theta = self.theta[:, batch_mask]
        self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (batch_theta.shape[0]*batch_theta.shape[1]) / 1e9

        # change phi into [:,w]
        if self.batch_loc[1] != 0:
            phi = batch_theta / self.norm_const
        else:
            # TODO move to cython ?
            small_mask = np.zeros(phi.shape[1], dtype=bool)
            mask_cnt = 0
            for i in xrange(batch_mask.shape[0]):
                if self.current_set[2][i]:
                    small_mask[mask_cnt] = batch_mask[i]
                    mask_cnt += 1
            phi = phi[:, small_mask]

        w_cnt = batch_theta.shape[1]

        Adk_mean, nkw_avg = self.sample_counts(train_cts, phi, self.batch_size, self.samples_per_update, self.samples,
                                               w_cnt, self.batch_map, self.batch_map_4w, MH_max, Adk=self.ndk,
                                               Adk_mean=self.ndk_avg); collect()
        # ******************************* update theta *********************************************
        (a, b, c) = self.step_size_params
        eps_t = a * (1 + self.update_ct / b) ** (-c)

        grad = self.beta - batch_theta + (self.D / self.batch_size) * (
            nkw_avg - np.sum(Adk_mean, 0)[:, np.newaxis] * phi); collect()
        if LWsampler:
            cul_time = time.time()
            g_theta_batch = g_theta[:, batch_mask]; collect()
            self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (g_theta_batch.shape[0]*g_theta_batch.shape[1]) / 1e9
            grad += - 2 * (batch_theta - g_theta_batch) / self.H ** 2; collect()
        stale = np.sum(batch_theta, 1)[:, np.newaxis]
        batch_theta[:, :] = np.abs(batch_theta + eps_t * grad +
                                  np.random.randn(batch_theta.shape[0], batch_theta.shape[1])
                                  * (2 * eps_t) ** .5 * batch_theta ** .5); collect()
        self.norm_const += np.sum(batch_theta, 1)[:, np.newaxis] - stale

        cul_time = time.time()
        self.theta[:, batch_mask] = batch_theta
        self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (batch_theta.shape[0]*batch_theta.shape[1]) / 1e9

        self.update_ct += 1

    # sapce: 24G
    def sample_counts(self, train_cts, phi, batch_D, num_samples, samples, w_cnt, batch_map, batch_map_4w,
                      MH_max, Adk=None, Adk_mean=None, perplexity=False):
        """ Adk, Adk_mean: input when in training, otherwise leave None
            batch_map_4w: samples contain 4w in training, thus the 4w_map is needed
            perplexity: if True, return only adk_mean"""
        batch_N = sum(len(doc) for doc in train_cts)
        z = [None for _ in xrange(batch_D)]
        if perplexity:
            Adk = np.zeros((batch_D, self.K), dtype=np.int32)
            Adk_mean = np.zeros(Adk.shape, dtype=np.float32)
        else:
            Adk.fill(0)
            Adk_mean.fill(0)

        nd = np.zeros(batch_D, dtype=np.int32)
        burn_in = num_samples // 2
        rand_kkk = np.random.randint(self.K, size=batch_N)

        if not perplexity:
            nkw = np.zeros((self.K, w_cnt), dtype=np.int32); collect()
            nkw_avg = np.zeros((self.K, w_cnt), dtype=np.float32)
            self.nk.fill(0)
            util_funcs.sample_z_par_alias(batch_D, train_cts, z, w_cnt, self.K, num_samples, burn_in, self.alpha,
                                          self.alpha_bar, self.beta, self.beta_bar, Adk, Adk_mean, nkw,
                                          nkw_avg, nd, self.nk, batch_map, batch_map_4w, phi, samples,
                                          MH_max, rand_kkk)
            nkw = None; collect()
            return Adk_mean, nkw_avg

        else:
            util_funcs.sample_z_par_alias_prplx(batch_D, train_cts, z, w_cnt, self.K, num_samples, burn_in, self.alpha,
                                                self.alpha_bar, self.beta, self.beta_bar, Adk, Adk_mean, nd, batch_map,
                                                batch_map_4w, phi, samples, MH_max, rand_kkk)
            return Adk_mean

    # space: 16G
    def gen_alias_table(self, MH_max, perplexity=False):
        # here the phi is [:,4w]
        if perplexity:
            phi = self.theta[:, self.test_doc[2]] / self.norm_const
            # samples has shape (w, 1e3 * I)
            samples = util_funcs.gen_obj(phi.shape[1])
            util_funcs.gen_alias_table(table_h=self.table_h, table_l=self.table_l, table_p=self.table_p,
                                       phi=phi / np.sum(phi, 0), batch_mask=self.test_doc[2], w_sample=self.test_doc[3],
                                       samples=samples, iter_per_update=self.samples_per_update, MH_max=MH_max)
        else:
            cul_time = time.time()
            tmp = self.theta[:, self.current_set[2]]
            self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (tmp.shape[0] * tmp.shape[1]) / 1e9
            phi = tmp / self.norm_const; tmp = None; collect()
            if self.w4_cnt is not None:
                util_funcs.kill_obj(self.w4_cnt, self.samples)
            self.w4_cnt = phi.shape[1]
            # samples has shape (w, 1e3 * I)
            samples = util_funcs.gen_obj(phi.shape[1])
            util_funcs.gen_batch_map(self.current_set[2], self.batch_map_4w, self.W)
            util_funcs.gen_alias_table(table_h=self.table_h, table_l=self.table_l, table_p=self.table_p,
                                       phi=phi / np.sum(phi, 0), batch_mask=self.current_set[2],
                                       w_sample=self.current_set[1], samples=samples,
                                       iter_per_update=self.samples_per_update, MH_max=MH_max)
        return samples, phi

    def get_perp_just_in_time(self, MH_max):
        """ the form of test_doc is: [ [[w], [..], ...], [[test_w], [..], ..], mask[], map[] ]"""
        samples, phi = self.gen_alias_table(MH_max, perplexity=True)
        util_funcs.gen_batch_map(self.test_doc[2], self.batch_map, self.W)
        Adk_mean = self.sample_counts(self.test_doc[0], phi, len(self.test_doc[0]), self.samples_per_update, samples,
                                      phi.shape[1], self.batch_map, self.batch_map, MH_max, perplexity=True); collect()
        Adk_mean += self.alpha
        Adk_mean /= np.sum(Adk_mean, 1)[:, np.newaxis]

        doc_len = len(self.test_doc[1])
        log_avg_probs = 0

        for d in xrange(doc_len):
            for w in self.test_doc[1][d]:
                log_avg_probs += np.log(np.dot(Adk_mean[d, :], phi[:, self.batch_map[w]]))

        num = sum([len(d) for d in self.test_doc[1]])
        util_funcs.kill_obj(phi.shape[1], samples)
        return np.exp(- log_avg_probs / num)

    def next_batch(self, MH_max):
        """ note: no detection if the file exist
            assuming format: [ [[d],[d],[d],[d]], map[], mask[], flag, [maskd[], maskd[], maskd[], maskd[]] ]"""
        if self.current_set is None:
            self.current_set = load(open(self.data_dir + 'saved_%i' % self.batch_loc[0], 'r'))
            self.samples, phi = self.gen_alias_table(MH_max)

            return self.current_set[0][0], phi

        elif self.batch_loc[1] < len(self.current_set[0]) - 1:
            self.batch_loc[1] += 1
            return self.current_set[0][self.batch_loc[1]], None

        else:
            if path.isfile(self.data_dir + 'saved_%i' % (self.batch_loc[0] + 1)):
                self.batch_loc[0] += 1
            else:
                print '************ worker %i fails to load set %i *************' % (self.rank, self.batch_loc[0])
                self.batch_loc[0] = 0
            self.batch_loc[1] = 0
            self.current_set = load(open(self.data_dir + 'saved_%i' % self.batch_loc[0], 'r'))
            self.samples, phi = self.gen_alias_table(MH_max)
            return self.current_set[0][0], phi


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


def run_very_large(MH_max, alpha=0.01, beta=0.0001, step_size_param=(0.01, 1000, 0.55)):
    num = 30
    train_set_size = 20726
    rank = 1
    doc_per_set = 200
    V = int(1e5)
    K = 10000
    jump = 10
    jump_bias = 0
    jump_hold = 0
    batch_size = 50
#    dir = '../corpus/b4_ff/'
    dir = '/home/lijm/WORK/yuan/b4_ff/'
    # TODO should be half, since gamma returns double
    max_len = 10000
    out_dir = '/home/lijm/WORK/yuan/'
#    out_dir = './'

    # num = 100
    # train_set_size = 5
    # offset = 0
    # doc_per_set = 15
    # V = 5
    # K = 5
    # jump = 10
    # jump_bias = 10
    # jump_hold = 0
    # batch_size = 5
    # # adjust step_size
    # step_size_param = (0.01, 5, 0.55)
    # dir = './small_test/'
    # max_len = 5

    output_name = out_dir + 'serial_perplexity' + time.strftime('_%m%d_%H%M%S', time.localtime()) + '.txt'

    sampler = LDSampler(0, dir, rank, train_set_size*doc_per_set, K, V, max_len, batch_size=batch_size, alpha=alpha,
                        beta=beta, epsilon=step_size_param[0], tau0=step_size_param[1], kappa=step_size_param[2])
    f = open(output_name, 'w')

    start_time = time.time()
    start_time = get_per(f, sampler, start_time)
    for i in xrange(num):
        print 'iter--->', i
        sampler.update(MH_max)

        if i < jump_bias and i != 0:
            start_time = get_per(f, sampler, start_time)
        elif (i + 1) % jump == 0 and (i + 1) >= jump_hold:
            start_time = get_per(f, sampler, start_time)

    f.close()


def get_per(f, sampler, start_time):
    start_time += sampler.time_bak; sampler.time_bak = 0
    per_s = time.time()
    print 'computing perplexity: '
    f.write('%.2f\t%.2f\n' % (sampler.get_perp_just_in_time(10), per_s - start_time))
    tmp = start_time + time.time() - per_s
    return tmp

from sys import argv
if __name__ == '__main__':
    # run_single()
    # test_on_alias()
    if len(argv) != 2:
        print 'try again'
    else:
        run_very_large(MH_max=int(argv[1]))

        # cProfile.runctx("run_very_large(10)", globals(), locals(), '/home/lijm/WORK/yuan/'+"Profile.prof")
        #
        # s = pstats.Stats('/home/lijm/WORK/yuan/'+"Profile.prof")
        # s.strip_dirs().sort_stats("time").print_stats()

