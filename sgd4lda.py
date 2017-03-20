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

"""
Before you read the code:

1. This whole file only defines the sgld sampler (with weierstrass kernel) object, after instantiating the object, you
will update the model using update function (see run_very_large() for an example)

2. the code is implemented straightforwardly from the paper, however there are few tricks that I use that is not reading
unfriendly: they are mainly for loading only part of the columns into the memory, since the whole word-topic matrix is
way too big. But our project assumes the whole matrix can fit into the memory, so you can ignore those tricks. I'll point
them out in the code, but here I will give a brief overview on what they are:
    a) collect(): just ignore it
    b) time, cul_time, time_bak: they are for counting the time, and you can ignore them
    c) h5py: I store mat in h5 format, but it has the same api as the numpy mat, so just ignore it
    d) mask: there are W-dim binary vectors to determine which slices of the topic-word mat should be loaded into memory
    e) map: this converts the W-dim vec into W'-dim vec, where the element in W' is the index of element with value True
        in W-dim vec.

3. for this file you mainly focus on update() and sample() functions, others more or less has more things to do with the
partial matrix loading.

"""

class LDSampler(object):
    # (max_len/10000) * 12 G
    def __init__(self, H, dir, rank, D, K, W, max_len, apprx, batch_size=50, alpha=0.01, beta=0.0001,
                 a=10**5.2, b=10**(-6), c=0.33, samples_per_update=50, test_doc='test_doc', suffix=None):
        """
        H: value sqrt(m)*sigma^(1+0.3)
        dir: indicates the root folder of each data folder, tmp file folder shall be created in here
        rank: indicate the subfolder where the docs exist
        D: the total docs in the local training set
        K: the num of topic
        W: the len of vocab
        max_len: maximum number of slices we can load into memory (ignore for 10708 prj)
        approx: a hack for counting the time (ignore for 10708 prj)
        samples_per_update = how many iterations needed to approximate the expectation term in SGLD update
        test_doc: file path of test documents
        time_bak: time used to load, correct with 1.5s; you need to set 0 every time you use it (ignore for 10708 prj)

        train_set: has form [ [[d],[d],[d],[d]], map[], mask[], flag, [maskd[], maskd[], maskd[], maskd[]] ]
        test_doc: [ [[w], [..], ...], [[test_w], [..], ..], mask[], map[] ]
        """

        # set the related parameters
        self.K = K
        self.batch_size = batch_size
        self.step_size_params = (a, b, c)
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
        self.dir = dir
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

        # comments for 10708 prj
        # here theta is the T matrix I refer to in the paper, and theta in the paper is named as g_theta (global theta)
        # I init theta one chunk at a time for the sake of memory
        # the norm_const is the normal constant of theta, where we have: phi (topic-word mat) = theta / norm_const
        start = 0
        while start < self.W:
            end = start + max_len
            end = end * (end <= self.W) + self.W * (end > self.W)
            tmp = np.random.gamma(1, 1, (self.K, end-start)); collect()
            self.theta[:, start:end] = tmp
            self.norm_const[:] += np.sum(tmp, 1)[:, np.newaxis]

            start = end
            tmp = None; collect()

        # ndk, nd and nk are cnt matrix similar to those used in vanilla collapsed gibbs sampling, and we use them
        # to count in every minibatches
        self.ndk = np.zeros((self.batch_size, K), dtype=np.int32)
        self.ndk_avg = np.zeros((self.batch_size, K), dtype=np.float32)
        self.nd = np.zeros(self.batch_size, dtype=np.int32)
        self.nk = np.zeros(K, dtype=np.int32)

        # these are alias tables for lightlda fast per-token sampling, you can refer to the summary in my paper to
        # refresh yourself
        self.table_h = np.zeros(self.K, dtype=np.int32)
        self.table_l = np.zeros(self.K, dtype=np.int32)
        self.table_p = np.zeros(self.K, dtype=np.float32)
        self.samples = None
        self.test_doc = load(open(self.data_dir + test_doc, 'r'))

        self.mask = np.ones(self.W, dtype=bool)

    # space: 20G
    def update(self, MH_max, LWsampler=False, g_theta=None, rec=None):
        """
            update the model with a minibatch

            input:
                MH_max: MH steps for stale alias table correction (please refer to AliasLDA and LightLDA paper for detail)
                LWsampler: are we running in distributed mode or not
                g_theta: the global theta (refer to as theta in my paper)
                rec: the mask indicates which columns are modified, which is used in distributed mode (for 10708 prj:
                as I said it's a tirck for loading part of the mat, you can pretend that you are loading the whole mat
                and ignore the details)
        """

        train_cts, phi = self.next_batch(MH_max, shift_dir=~LWsampler); collect()

        batch_mask = self.current_set[4][self.batch_loc[1]]
        if LWsampler: rec[:] = rec + batch_mask
        # generate the map from binary vec to index vec (ignore for 10708 prj)
        util_funcs.gen_batch_map(batch_mask, self.batch_map, self.W)

        cul_time = time.time()
        batch_theta = self.theta[:, batch_mask]
        self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (batch_theta.shape[0]*batch_theta.shape[1]) / 1e9

        # change phi into [:,w], where w is the number of words used in minibatch, which is usually w << W
        if self.batch_loc[1] != 0:
            phi = batch_theta / self.norm_const
        else:
            small_mask = np.zeros(phi.shape[1], dtype=bool)
            mask_cnt = 0
            for i in xrange(batch_mask.shape[0]):
                if self.current_set[2][i]:
                    small_mask[mask_cnt] = batch_mask[i]
                    mask_cnt += 1
            phi = phi[:, small_mask]

        w_cnt = batch_theta.shape[1]

        # compute the expectation term in SGLD
        Adk_mean, nkw_avg = self.sample_counts(train_cts, phi, self.batch_size, self.samples_per_update, self.samples,
                                               w_cnt, self.batch_map, self.batch_map_4w, MH_max, Adk=self.ndk,
                                               Adk_mean=self.ndk_avg)

        # ******************************* gradient decent on theta (T in paper) ***************************************
        (a, b, c) = self.step_size_params
        eps_t = (a + self.update_ct / b) ** (-c)

        grad = self.beta - batch_theta + (self.D / self.batch_size) * (
            nkw_avg - np.sum(Adk_mean, 0)[:, np.newaxis] * phi)

        # if in distributed mode then we add the gradient from the weierstrass kernel
        if LWsampler:
            cul_time = time.time()
            g_theta_batch = g_theta[:, batch_mask]; collect()
            self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (self.K*w_cnt) / 1e9
            grad += - 2 * (batch_theta - g_theta_batch) / self.H ** 2

        stale = np.sum(batch_theta, 1)[:, np.newaxis]
        batch_theta[:, :] = np.abs(batch_theta + eps_t * grad + np.random.randn(self.K, w_cnt)*(2*eps_t)**.5 * batch_theta**.5)
        self.norm_const += np.sum(batch_theta, 1)[:, np.newaxis] - stale

        cul_time = time.time()
        self.theta[:, batch_mask] = batch_theta
        self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (self.K*w_cnt) / 1e9

        self.update_ct += 1

    # sapce: 24G
    def sample_counts(self, train_cts, phi, batch_D, num_samples, samples, w_cnt, batch_map, batch_map_4w,
                      MH_max, Adk=None, Adk_mean=None, perplexity=False):
        """
            compute the expectation term in SGLD which is essentially the collapsed gibbs sampling given the minibatch
            data. Here, instead of using vanilla per-token sampling, we use LightLDA per-token sampling. This is
            implemented using cython for performance, one should refer to util_func.pyx for detail.

            input:
                train_cts: minibatch data
                phi: topic-word mat
                batch_D: batch size
                num_samples: num of iterations for expectation estimation
                w_cnt: how many unique words in this batch
                Adk, Adk_mean: input when in training, otherwise leave None
                batch_map, batch_map_4w: maps (samples contain 4w in training, thus the 4w_map is needed)
                perplexity: if True, return only adk_mean
        """
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
    def gen_alias_table(self, MH_max, theta, norm_const, perplexity=False):
        """
            generate alias table for fast per-token sampling
        """
        # here the phi is [:,4w]
        if perplexity:
            phi = theta[:, self.test_doc[2]] / norm_const
            # samples has shape (w, 1e3 * I)
            samples = util_funcs.gen_obj(phi.shape[1])
            util_funcs.gen_alias_table(table_h=self.table_h, table_l=self.table_l, table_p=self.table_p,
                                       phi=phi / np.sum(phi, 0), batch_mask=self.test_doc[2], w_sample=self.test_doc[3],
                                       samples=samples, iter_per_update=self.samples_per_update, MH_max=MH_max)
        else:
            cul_time = time.time()
            tmp = theta[:, self.current_set[2]]
            self.time_bak += time.time() - cul_time - 1.5 * self.apprx * (tmp.shape[0] * tmp.shape[1]) / 1e9
            phi = tmp / norm_const; tmp = None; collect()
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

    def get_perp_just_in_time(self, MH_max, theta=None, norm_const=None):
        """
            note:
                assume the form of test_doc is: [ [[w], [..], ...], [[test_w], [..], ..], mask[], map[] ]
        """
        theta = self.theta if theta is None else theta; norm_const = self.norm_const if norm_const is None else norm_const
        samples, phi = self.gen_alias_table(MH_max, theta, norm_const, perplexity=True)
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

    def next_batch(self, MH_max, shift_dir):
        """
            load the next batch of documents and return slices of phi that are involved

            note:
                no detection if the file exist
                we assuming format: [ [[d],[d],[d],[d]], map[], mask[], flag, [maskd[], maskd[], maskd[], maskd[]] ]

            input:
                shift_dir: if true, we will jump to next folder if file here is exhausted
        """
        if self.current_set is None:
            self.current_set = load(open(self.data_dir + 'saved_%i' % self.batch_loc[0], 'r'))
            self.samples, phi = self.gen_alias_table(MH_max, self.theta, self.norm_const)

            return self.current_set[0][0], phi

        elif self.batch_loc[1] < len(self.current_set[0]) - 1:
            self.batch_loc[1] += 1
            return self.current_set[0][self.batch_loc[1]], None

        else:
            if path.isfile(self.data_dir + 'saved_%i' % (self.batch_loc[0] + 1)):
                self.batch_loc[0] += 1
            else:
                print '************ worker %i fails to load set %i *************' % (self.rank, self.batch_loc[0])
                if shift_dir and path.isdir(self.dir + str(self.rank+1) + '/'):
                    self.rank += 1
                    self.data_dir = self.dir + str(self.rank) + '/'
                    print '************ shift to folder %s *************' % self.data_dir
                elif shift_dir:
                    self.rank = 1
                    self.data_dir = self.dir + str(self.rank) + '/'
                    print '************ shift to folder %s *************' % self.data_dir
                self.batch_loc[0] = 0
            self.batch_loc[1] = 0
            self.current_set = load(open(self.data_dir + 'saved_%i' % self.batch_loc[0], 'r'))

            # for 10708 prj: note that I obtain phi here, since for each minibatch only part of the phi is involved
            # if you can hold whole phi in memory then you don't need to do it here.
            self.samples, phi = self.gen_alias_table(MH_max, self.theta, self.norm_const)
            return self.current_set[0][0], phi

# for 10708 just ignore
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


def run_very_large(MH_max, alpha=0.01, beta=0.0001, step_size_param=(10**5.2, 10**(-6), 0.33)):
    num = 12000
    train_set_size = 20726
    rank = 1
    doc_per_set = 200
    V = int(1e5)
    K = 100
    jump = 50
    jump_bias = 0
    jump_hold = 0
    batch_size = 50
    dir = '../corpus/b4_ff/'
    out_dir = './'
    # dir = '/home/lijm/WORK/yuan/b4_ff/'
    # out_dir = '/home/lijm/WORK/yuan/'
    max_len = 10000

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

    sampler = LDSampler(0, dir, rank, train_set_size*doc_per_set, K, V, max_len, 1, batch_size=batch_size, alpha=alpha,
                        beta=beta, a=step_size_param[0], b=step_size_param[1], c=step_size_param[2])

    start_time = get_per(output_name, sampler, time.time())
    for i in xrange(num):
        print 'iter--->', i
        sampler.update(MH_max)

        if i < jump_bias and i != 0:
            start_time = get_per(output_name, sampler, start_time)
        elif (i + 1) % jump == 0 and (i + 1) >= jump_hold:
            start_time = get_per(output_name, sampler, start_time)


def get_per(output_name, sampler, start_time):
    start_time += sampler.time_bak; sampler.time_bak = 0
    per_s = time.time()

    print 'computing perplexity: '
    prplx = sampler.get_perp_just_in_time(10)

    print 'perplexity: %.2f' % prplx
    f = open(output_name, 'a')
    f.write('%.2f\t%.2f\n' % (prplx, per_s - start_time))
    f.close()

    return start_time + time.time() - per_s


if __name__ == '__main__':

    run_very_large(MH_max=10)

    # cProfile.runctx("run_very_large(10)", globals(), locals(), '/home/lijm/WORK/yuan/'+"Profile.prof")
    #
    # s = pstats.Stats('/home/lijm/WORK/yuan/'+"Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()



