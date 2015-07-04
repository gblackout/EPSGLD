#cython: profile=False

from __future__ import division
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport RAND_MAX, rand, srand
from libc.time cimport time, clock
from libc.stdlib cimport malloc, free
ctypedef np.float64_t dtype_t
ctypedef np.uint32_t uitype_t
from sys import stdout
from os import listdir
import re

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_par_alias(int doc_size,
                 list doc_dicts,
                 list z,
                 int W,
                 int K,
                 int num_samples,
                 int burn_in,
                 float alpha,
                 float alpha_bar,
                 float beta,
                 float beta_bar,
                 np.ndarray[np.int32_t, ndim=2] Adk,
                 np.ndarray[np.float32_t, ndim=2] Adk_avg,
                 np.ndarray[np.int32_t, ndim=2] Bkw,
                 np.ndarray[np.float32_t, ndim=2] Bkw_avg,
                 np.ndarray[np.int32_t, ndim=1] nd,
                 np.ndarray[np.int32_t, ndim=1] nk,
                 np.ndarray[np.int32_t, ndim=1] batch_map,
                 np.ndarray[np.int32_t, ndim=1] batch_map_4w,
                 np.ndarray[np.float32_t, ndim=2] phi,
                 Samples samples,
                 int MH_max,
                 np.ndarray[np.long_t, ndim=1] rand_kkk,
                 # np.ndarray[np.int32_t, ndim=1] rand_kkk,
                 ):
        """ note that doc_size is not batch_size, since the test set size may be not consistent
            W here is the same as phi.shape[1], i.e. the w_cnt"""
        cdef Py_ssize_t d, doc_len, i, zOld, w, k, zNew, sim, zInit
        cdef np.ndarray[np.int32_t, ndim=1] z_list
        cdef bint tic_toc = True
        cdef int rand_max = <int>RAND_MAX
        cdef Py_ssize_t uni_idx = 0

        for d in range(doc_size):
            doc_len = len(doc_dicts[d])
            z[d] = np.zeros(doc_len, dtype=np.int32)
            z_list = z[d]
            for i in xrange(doc_len):
                w = batch_map[doc_dicts[d][i]]

                zInit = rand_kkk[uni_idx]

                Bkw[zInit, w] += 1
                Adk[d, zInit] += 1
                nk[zInit] += 1
                nd[d] += 1
                z_list[i] = zInit

                uni_idx += 1
        for sim in xrange(num_samples):

            for d in xrange(doc_size):
                doc_len = len(doc_dicts[d])
                z_list = z[d]

                for i in xrange(doc_len):
                    w = batch_map[doc_dicts[d][i]]

                    zOld = z_list[i]
                    Adk[d, zOld] -= 1
                    Bkw[zOld,w] -= 1
                    nd[d] -= 1
                    nk[zOld] -= 1

                    zNew = sample_alias_table_inner(MH_max, tic_toc, zOld, w, batch_map_4w[doc_dicts[d][i]], d,
                                                    &z_list[0], &Adk[0,0], &phi[0,0], samples, alpha_bar, alpha,
                                                    beta, beta_bar, K, W, rand_max, doc_len)

                    z_list[i] = zNew
                    Adk[d, zNew] += 1
                    Bkw[zNew, w] += 1
                    nd[d] += 1
                    nk[zOld] += 1

                    tic_toc = not tic_toc

            if sim >= burn_in:
                Adk_avg += Adk
                Bkw_avg += Bkw

        Adk_avg /= (num_samples - burn_in)
        Bkw_avg /= (num_samples - burn_in)


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_par_alias_prplx(int doc_size,
                 list doc_dicts,
                 list z,
                 int W,
                 int K,
                 int num_samples,
                 int burn_in,
                 float alpha,
                 float alpha_bar,
                 float beta,
                 float beta_bar,
                 np.ndarray[np.int32_t, ndim=2] Adk,
                 np.ndarray[np.float32_t, ndim=2] Adk_avg,
                 np.ndarray[np.int32_t, ndim=1] nd,
                 np.ndarray[np.int32_t, ndim=1] batch_map,
                 np.ndarray[np.int32_t, ndim=1] batch_map_4w,
                 np.ndarray[np.float32_t, ndim=2] phi,
                 Samples samples,
                 int MH_max,
                 np.ndarray[np.long_t, ndim=1] rand_kkk,
                 # np.ndarray[np.int32_t, ndim=1] rand_kkk,
                 ):
        """ note that doc_size is not batch_size, since the test set size may be not consistent
            W here is the same as phi.shape[1], i.e. the w_cnt"""
        cdef Py_ssize_t d, doc_len, i, zOld, w, k, zNew, sim, zInit
        cdef np.ndarray[np.int32_t, ndim=1] z_list
        cdef bint tic_toc = True
        cdef int rand_max = <int>RAND_MAX
        cdef Py_ssize_t uni_idx = 0

        for d in range(doc_size):
            doc_len = len(doc_dicts[d])
            z[d] = np.zeros(doc_len, dtype=np.int32)
            z_list = z[d]
            for i in xrange(doc_len):
                w = batch_map[doc_dicts[d][i]]

                zInit = rand_kkk[uni_idx]

                Adk[d, zInit] += 1
                nd[d] += 1
                z_list[i] = zInit

                uni_idx += 1
        for sim in xrange(num_samples):

            for d in xrange(doc_size):
                doc_len = len(doc_dicts[d])
                z_list = z[d]

                for i in xrange(doc_len):
                    w = batch_map[doc_dicts[d][i]]

                    zOld = z_list[i]
                    Adk[d, zOld] -= 1
                    nd[d] -= 1

                    zNew = sample_alias_table_inner(MH_max, tic_toc, zOld, w, batch_map_4w[doc_dicts[d][i]], d,
                                                    &z_list[0], &Adk[0,0], &phi[0,0], samples, alpha_bar, alpha,
                                                    beta, beta_bar, K, W, rand_max, doc_len)
                    z_list[i] = zNew
                    Adk[d, zNew] += 1
                    nd[d] += 1

                    tic_toc = not tic_toc

            if sim >= burn_in:
                Adk_avg += Adk

        Adk_avg /= (num_samples - burn_in)


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_par(int doc_size,
                 list doc_dicts,
                 list z,
                 int K,
                 int num_samples,
                 int burn_in,
                 float alpha,
                 np.ndarray[np.int32_t, ndim=2] Adk,
                 np.ndarray[np.float32_t, ndim=2] Adk_avg,
                 np.ndarray[np.int32_t, ndim=2] Bkw,
                 np.ndarray[np.float32_t, ndim=2] Bkw_avg,
                 np.ndarray[np.int32_t, ndim=1] batch_map,
                 np.ndarray[np.float32_t, ndim=2] phi,
                 np.ndarray[np.float64_t, ndim=1] uni_rvs,
                 np.ndarray[np.long_t, ndim=1] rand_kkk):
                 # np.ndarray[np.int32_t, ndim=1] rand_kkk):
        """ note that doc_size is not batch_size, since the test set size may be not consistent"""
        cdef np.ndarray[np.float64_t, ndim=1] cumprobs = np.zeros(K, dtype=np.float64)
        cdef Py_ssize_t d, doc_len, i, zOld, w, k, rc_start, rc_stop, rc_mid, zNew, sim
        cdef double prob_sum, uni_rv
        cdef np.ndarray[np.int32_t, ndim=1] z_list
        cdef int uni_idx = 0

        for d in range(doc_size):
            doc_len = len(doc_dicts[d])
            z[d] = np.zeros(doc_len, dtype=np.int32)
            z_list = z[d]
            for i in xrange(doc_len):
                w = batch_map[doc_dicts[d][i]]

                zInit = rand_kkk[uni_idx]

                Bkw[zInit, w] += 1
                Adk[d, zInit] += 1
                z_list[i] = zInit

                uni_idx += 1

        uni_idx = 0

        for sim in xrange(num_samples):

            for d in xrange(doc_size):
                doc_len = len(doc_dicts[d])
                z_list = z[d]

                for i in xrange(doc_len):
                    zOld = z_list[i]
                    w = batch_map[doc_dicts[d][i]]

                    prob_sum = 0
                    for k in range(K):
                        cumprobs[k] = prob_sum
                        prob_sum += (alpha + Adk[d, k] - (k == zOld)) * phi[k, w]

                    uni_rv = prob_sum * uni_rvs[uni_idx]
                    uni_idx += 1

                    rc_start = 0
                    rc_stop = K
                    while rc_start < rc_stop - 1:
                        rc_mid = (rc_start + rc_stop) // 2
                        if cumprobs[rc_mid] <= uni_rv:
                            rc_start = rc_mid
                        else:
                            rc_stop = rc_mid
                    zNew = rc_start

                    z_list[i] = zNew
                    Adk[d, zOld] -= 1
                    Adk[d, zNew] += 1
                    Bkw[zOld, w] -= 1
                    Bkw[zNew, w] += 1

            if sim >= burn_in:
                Adk_avg += Adk
                Bkw_avg += Bkw

        Adk_avg /= (num_samples - burn_in)
        Bkw_avg /= (num_samples - burn_in)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int sample_alias_table_inner(int MH_max, bint word_pro, int old, int w, int w_samples, int dt,
                              np.int32_t* z_list,
                              np.int32_t* ndk,
                              np.float32_t* phi,
                              Samples samples,
                              float alpha_bar,float alpha, float beta, float beta_bar,
                              int K,
                              int W,
                              int rand_max,
                              int doc_len):
    """ w is actually the batch_map[w]
        W here is the same as phi.shape[1], i.e. the w_cnt"""

    cdef int k_old = old
    cdef int k_new, i_u
    cdef double const_rate, const_down
    cdef double acc = 0
    cdef int* zPro
    cdef int cur_ptr

    # ******************************* gen zPro sequence *********************************************
    zPro = <int*>malloc(MH_max*sizeof(int))
    if word_pro:
        cur_ptr = samples.w_pointer[w_samples]
        for i_u in xrange(MH_max):
            zPro[i_u] = samples.w_sample[w_samples][cur_ptr+i_u]
        samples.w_pointer[w_samples] += MH_max
    else:
        for i_u in xrange(MH_max):
            if rand_u() * (doc_len + alpha_bar) > doc_len:
                zPro[i_u] = rand_k_inner(K, rand_max)
            else:
                zPro[i_u]= z_list[rand_k_inner(doc_len, rand_max)]
    # ******************************* get acc_rate for zPro *********************************************
    for i_m in xrange(MH_max):
        k_new = zPro[i_m]

        const_rate = (ndk[dt*K+k_new] + alpha) / (ndk[dt*K+k_old] + alpha)
        if word_pro:
            acc = const_rate
        else:
            const_rate *= (phi[k_new*W+w] / phi[k_old*W+w])
            acc = const_rate * (ndk[dt*K+k_old] + alpha+ (k_old == old)) / (ndk[dt*K+k_new] + alpha + (k_new == old))

        if rand_u() < min(1, acc):
            k_old = k_new
    free(zPro)
    return k_old


@cython.boundscheck(False)
@cython.wraparound(False)
def gen_alias_table(np.ndarray[np.int32_t, ndim=1] table_h,
                    np.ndarray[np.int32_t, ndim=1] table_l,
                    np.ndarray[np.float32_t, ndim=1] table_p,
                    np.ndarray[np.float32_t, ndim=2] phi,
                    np.ndarray[np.uint8_t, ndim=1, cast=True] batch_mask,
                    np.ndarray[np.int32_t, ndim=1] w_sample,
                    Samples samples,
                    int iter_per_update,
                    int MH_max):
        """ the batch_mask is used to indicate columns that are used in this 4-batch"""
        cdef int V = phi.shape[1], K = phi.shape[0]

        cdef np.ndarray[np.float32_t, ndim=1] low = np.zeros(K, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1] high = np.zeros(K, dtype=np.float32)
        cdef np.ndarray[np.int32_t, ndim=1] low_map = np.zeros(K, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] high_map = np.zeros(K, dtype=np.int32)

        cdef int i_p, i_l, zero_cnt, i_v, ls, le, hs, he, table_cnt
        cdef float tmp, avg, after

        cdef int cnt2w_ind = 0

        avg = 1 / (<float>K)
        for i_v in xrange(V):
            table_cnt = hs = ls = he = le = 0
            for i_l in xrange(K):
                if phi[i_l, i_v] - avg > -1e-8:
                    high[he] = phi[i_l, i_v]
                    high_map[he] = i_l
                    he += 1
                else:
                    low[le] = phi[i_l, i_v]
                    low_map[le] = i_l
                    le += 1

            while ls != le:

                table_h[table_cnt] = high_map[hs]
                table_l[table_cnt] = low_map[ls]
                table_p[table_cnt] = low[ls]

                table_cnt += 1
                after = high[hs] + low[ls] - avg
                ls += 1
                high[hs] = after

                while (high[hs] - avg <= -1e-8) and (hs < he-1):
                    table_l[table_cnt] = high_map[hs]
                    table_p[table_cnt] = high[hs]
                    hs += 1
                    table_h[table_cnt] = high_map[hs]
                    table_cnt += 1
                    high[hs] += high[hs-1] - avg

            while hs != he:
                table_h[table_cnt] = table_l[table_cnt] = high_map[hs]
                table_p[table_cnt] = high[hs]
                hs += 1
                table_cnt += 1

            while not batch_mask[cnt2w_ind]:
                cnt2w_ind += 1
            samples.w_sample[i_v] = sample_alias_batch(w_sample[cnt2w_ind]*iter_per_update*MH_max, &table_h[0], &table_l[0], &table_p[0], K)
            cnt2w_ind += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int* sample_alias_batch(int num,
                       np.int32_t* table_h,
                       np.int32_t* table_l,
                       np.float32_t* table_p,
                       int K):
    """ table: [[low_index, high_index, low_p], [...], ...]"""
    cdef int i_n, p
    cdef double avg = 1 / (<double>K), runi
    cdef int* samples_row = <int*>malloc(num*sizeof(int))

    for i_n in xrange(num):
        runi = rand_u()
        p = int(K * runi)
        if runi - p * avg < table_p[p]:
            samples_row[i_n] = table_l[p]
        else:
            samples_row[i_n] = table_h[p]
    return samples_row

@cython.boundscheck(False)
@cython.wraparound(False)
def gen_batch_map(np.ndarray[np.uint8_t, ndim=1, cast=True] mask,
                  np.ndarray[np.int32_t, ndim=1] map,
                  int W):
    """ NOTICE: W idicates the global W"""
    cdef int cnt = 0
    cdef int i

    for i in xrange(W):
        if mask[i] is True:
            map[i] = cnt
            cnt += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def init_light(list train_cts,
               np.ndarray[np.int32_t, ndim=2] ndk,
               np.ndarray[np.int32_t, ndim=2] nkw,
               np.ndarray[np.int32_t, ndim=1] nd,
               np.ndarray[np.int32_t, ndim=1] nk,
               int doc_per_set,
               np.ndarray[object, ndim=1] z,
               int K,
               int start,
               int end,
               bint z_init):

    cdef Py_ssize_t d, w, i, doc_len, zInit
    cdef np.ndarray[np.int32_t, ndim=1] z_list

    for d in range(doc_per_set):
        doc_len = len(train_cts[d])
        if z_init: z[d] = np.zeros(doc_len, dtype=np.int32)
        z_list = z[d]
        for i in xrange(doc_len):
            w = train_cts[d][i]

            if not (start <= w < end): continue

            zInit = rand_k(K)
            ndk[d,zInit] += 1
            nkw[zInit,w-start] += 1
            nk[zInit] += 1
            nd[d] += 1
            z_list[i] = zInit

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sample_light(int K,
                 float alpha,
                 float alpha_bar,
                 float beta,
                 float beta_bar,
                 np.ndarray[np.int32_t, ndim=2] table_h,
                 np.ndarray[np.int32_t, ndim=2] table_l,
                 np.ndarray[np.float32_t, ndim=2] table_p,
                 np.ndarray[np.int32_t, ndim=2] nkw_part,
                 np.ndarray[np.int32_t, ndim=1] nk,
                 np.ndarray[np.int32_t, ndim=2] ndk,
                 np.ndarray[np.int32_t, ndim=1] nd,
                 np.ndarray[np.int32_t, ndim=2] nkw_stale,
                 np.ndarray[np.int32_t, ndim=1] nk_stale,
                 np.ndarray[object, ndim=1] z,
                 list train_cts,
                 int MH_max,
                 int part_0,
                 int part_1,
                 int doc_per_set,
                 char* file_name):
        """ note that doc_size is not batch_size, since the test set size may be not consistent
            W here is the same as phi.shape[1], i.e. the w_cnt"""
        cdef Py_ssize_t doc_len, i, zOld, w, k, zNew, i_u, i_m, dt, w_part
        cdef np.ndarray[np.int32_t, ndim=1] z_list
        cdef bint tic_toc = True
        cdef int rand_max = <int>RAND_MAX
        cdef int k_old, k_new
        cdef double const_rate, acc = 0
        cdef int* zPro

        for dt in xrange(doc_per_set):
            doc_len = len(train_cts[dt])
            z_list = z[dt]

            for i in xrange(doc_len):
                w = train_cts[dt][i]

                if not (part_0 <= w < part_1): continue

                zOld = z_list[i]
                w_part = w - part_0
                nkw_part[zOld, w_part] -= 1
                nk[zOld] -= 1
                ndk[dt, zOld] -= 1
                nd[dt] -= 1

                # ******************************* gen zPro sequence *********************************************
                k_old = zOld
                if tic_toc:
                    zPro = sample_alias(MH_max, &table_h[w_part, 0], &table_l[w_part, 0], &table_p[w_part, 0], K)
                else:
                    zPro = <int*>malloc(MH_max*sizeof(int))
                    for i_u in xrange(MH_max):
                        if rand_u() * (nd[dt] + 1 + alpha_bar) > nd[dt] + 1:
                            zPro[i_u] = rand_k_inner(K, rand_max)
                        else:
                            zPro[i_u] = z_list[rand_k_inner(nd[dt] + 1, rand_max)]
                # ******************************* get acc_rate for zPro *********************************************
                for i_m in xrange(MH_max):
                    k_new = zPro[i_m]

                    const_rate = (ndk[dt, k_new] + alpha) * \
                                 (nkw_part[k_new, w_part] + beta) * \
                                 (nk[k_old] + beta_bar) / \
                                 ((nk[k_new] + beta_bar) *
                                  (ndk[dt, k_old] + alpha) *
                                  (nkw_part[k_old, w_part] + beta))
                    if tic_toc:
                        acc = const_rate * (nkw_stale[k_old, w_part] + beta) * (nk_stale[k_new] + beta_bar) /  \
                              ((nkw_stale[k_new, w_part] + beta) * (nk_stale[k_old] + beta_bar))
                    else:
                        acc = const_rate * (ndk[dt, k_old] + alpha + (k_old == zOld)) / \
                              (ndk[dt, k_new] + alpha + (k_old == zOld))

                    if rand_u() < min(1, acc):
                        k_old = k_new

                free(zPro)
                k = k_old

                z_list[i] = k
                nkw_part[k, w_part] += 1
                nk[k] += 1
                ndk[dt, k] += 1
                nd[dt] += 1

                tic_toc = ~tic_toc



@cython.boundscheck(False)
@cython.wraparound(False)
def sample_alias_table(int MH_max, bint word_pro, int old, int w, int w_part, int dt, np.ndarray[np.int32_t, ndim=1] z_list,
                       np.ndarray[np.int32_t, ndim=1] nd,
                       np.ndarray[np.int32_t, ndim=1] nk,
                       np.ndarray[np.int32_t, ndim=1] nk_stale,
                       np.ndarray[np.int32_t, ndim=2] ndk,
                       np.ndarray[np.int32_t, ndim=2] nkw,
                       np.ndarray[np.int32_t, ndim=2] nkw_stale,
                       float alpha_bar,float alpha, float beta, float beta_bar,
                       int K,
                       np.ndarray[np.int32_t, ndim=1] table_h,
                       np.ndarray[np.int32_t, ndim=1] table_l,
                       np.ndarray[np.float32_t, ndim=1] table_p):
    """ this is different from the original one since the zPro is generated outside"""

    cdef int k_old = old, k_new
    cdef double const_rate, acc = 0
    cdef np.ndarray[np.float64_t, ndim=2] unis = np.random.rand(2, MH_max)
    # cdef np.ndarray[np.int32_t, ndim=1] unis_k = np.random.randint(K, size=MH_max)
    # cdef np.ndarray[np.int32_t, ndim=1] unis_d = np.random.randint(nd[dt] + 1, size=MH_max)
    cdef np.ndarray[np.long_t, ndim=1] unis_k = np.random.randint(K, size=MH_max)
    cdef np.ndarray[np.long_t, ndim=1] unis_d = np.random.randint(nd[dt] + 1, size=MH_max)
    cdef int* zPro

    # ******************************* gen zPro sequence *********************************************
    if word_pro:
        zPro = sample_alias(MH_max, &table_h[0], &table_l[0], &table_p[0], K)
    else:
        zPro = <int*>malloc(MH_max*sizeof(int))
        for i_u in xrange(MH_max):
            if unis[0, i_u] * (nd[dt] + 1 + alpha_bar) > nd[dt] + 1:
                zPro[i_u] = unis_k[i_u]
            else:
                zPro[i_u] = z_list[unis_d[i_u]]
    # ******************************* get acc_rate for zPro *********************************************
    for i_m in xrange(MH_max):
        k_new = zPro[i_m]
        const_rate = (ndk[dt, k_new] + alpha) * \
                     (nkw[k_new, w_part] + beta) * \
                     (nk[k_old] + beta_bar) / \
                     ((nk[k_new] + beta_bar) *
                      (ndk[dt, k_old] + alpha) *
                      (nkw[k_old, w_part] + beta))
        if word_pro:
            acc = const_rate * (nkw_stale[k_old, w_part] + beta) * (nk_stale[k_new] + beta_bar) /  \
                  ((nkw_stale[k_new, w_part] + beta) * (nk_stale[k_old] + beta_bar))
        else:
            acc = const_rate * (ndk[dt, k_old] + alpha + (k_old == old)) / (ndk[dt, k_new] + alpha + (k_old == old))

        if unis[1, i_m] < min(1, acc):
            k_old = k_new

    free(zPro)
    return k_old


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_gibbs(
                 np.ndarray[np.int32_t, ndim=2] ndk,
                 np.ndarray[np.int32_t, ndim=2] nkw,
                 np.ndarray[np.int32_t, ndim=1] nk,
                 float alpha,
                 float beta,
                 float beta_bar,
                 int d,
                 int w,
                 int w_part,
                 int K):

    cdef Py_ssize_t k
    cdef Py_ssize_t rc_start = 0, rc_mid, rc_stop = K
    cdef double prob_sum, uni_rv
    cdef np.ndarray[np.float64_t, ndim=1] cumprobs = np.zeros(K, dtype=np.float64)

    prob_sum = 0
    for k in range(K):
        cumprobs[k] = prob_sum
        prob_sum +=  (alpha + ndk[d,k]) * (nkw[k, w_part] + beta) / (nk[k] + beta_bar)
    uni_rv = prob_sum  * rand_u()

    rc_start = 0
    rc_stop  = K
    while rc_start < rc_stop - 1:
        rc_mid = (rc_start + rc_stop) // 2
        if cumprobs[rc_mid] <= uni_rv:
            rc_start = rc_mid
        else:
            rc_stop = rc_mid

    return rc_start


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int* sample_alias(int num,
                       np.int32_t* table_h,
                       np.int32_t* table_l,
                       np.float32_t* table_p,
                       int K):
    """ table: [[low_index, high_index, low_p], [...], ...]"""
    cdef int i_n, p
    cdef double avg = 1 / (<double>K), runi
    cdef int* sample = <int*>malloc(num*sizeof(int))

    for i_n in xrange(num):
        runi = rand_u()
        p = int(K * runi)
        if runi - p * avg < table_p[p]:
            sample[i_n] = table_l[p]
        else:
            sample[i_n] = table_h[p]

    return sample


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_par_alias_per(int doc_size,
                 list doc_dicts,
                 list z,
                 int W,
                 int K,
                 int num_samples,
                 int burn_in,
                 float alpha,
                 float alpha_bar,
                 float beta,
                 float beta_bar,
                 np.ndarray[np.int32_t, ndim=2] Adk,
                 np.ndarray[np.float32_t, ndim=2] Adk_avg,
                 np.ndarray[np.int32_t, ndim=1] batch_map,
                 np.ndarray[np.float32_t, ndim=2] phi,
                 Samples samples,
                 int MH_max,
                 np.ndarray[np.long_t, ndim=1] rand_kkk,
                 # np.ndarray[np.int32_t, ndim=1] rand_kkk,
                 np.ndarray[np.uint8_t, ndim=1, cast=True] w_mask,
                 bint no_sum=False):
        """ note that doc_size is not batch_size, since the test set size may be not consistent
            W here is the same as phi.shape[1], i.e. the w_cnt"""
        cdef Py_ssize_t d, doc_len, i, zOld, w, k, zNew, sim, zInit
        cdef np.ndarray[np.int32_t, ndim=1] z_list
        cdef bint tic_toc = True
        cdef int rand_max = <int>RAND_MAX
        cdef Py_ssize_t uni_idx = 0

        for d in range(doc_size):
            doc_len = len(doc_dicts[d])
            z[d] = np.zeros(doc_len, dtype=np.int32)
            z_list = z[d]
            for i in xrange(doc_len):
                w = doc_dicts[d][i]
                if not w_mask[w]:
                    continue
                w = batch_map[w]

                zInit = rand_kkk[uni_idx]

                Adk[d, zInit] += 1
                z_list[i] = zInit

                uni_idx += 1

        for sim in xrange(num_samples):

            for d in xrange(doc_size):
                doc_len = len(doc_dicts[d])
                z_list = z[d]

                for i in xrange(doc_len):
                    w = doc_dicts[d][i]
                    if not w_mask[w]:
                        continue
                    w = batch_map[w]

                    zOld = z_list[i]
                    Adk[d, zOld] -= 1

                    zNew = sample_alias_table_inner(MH_max, tic_toc, zOld, w, w, d, &z_list[0], &Adk[0,0], &phi[0,0],
                                                    samples, alpha_bar, alpha, beta, beta_bar, K, W, rand_max, doc_len)
                    z_list[i] = zNew
                    Adk[d, zNew] += 1

                    tic_toc = not tic_toc

            if sim >= burn_in:
                Adk_avg += Adk

        Adk_avg /= (num_samples - burn_in)


@cython.boundscheck(False)
@cython.wraparound(False)
def gen_alias_table_light(np.ndarray[np.int32_t, ndim=2] table_h,
                    np.ndarray[np.int32_t, ndim=2] table_l,
                    np.ndarray[np.float32_t, ndim=2] table_p,
                    np.ndarray[np.float32_t, ndim=2] phi,
                    np.ndarray[np.uint8_t, ndim=1, cast=True] w_mask):
        """ the w_mask is used to indicate columns that are not met for this corpus"""
        cdef int V = phi.shape[1], K = phi.shape[0]

        cdef np.ndarray[np.float32_t, ndim=1] low = np.zeros(K, dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=1] high = np.zeros(K, dtype=np.float32)
        cdef np.ndarray[np.int32_t, ndim=1] low_map = np.zeros(K, dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] high_map = np.zeros(K, dtype=np.int32)

        cdef int i_p, i_l, zero_cnt, i_v, ls, le, hs, he, table_cnt
        cdef float tmp, avg, after

        # debug
        cdef bint test = True

        avg = 1 / (<float>K)
        for i_v in xrange(V):

            if not w_mask[i_v]: continue

            table_cnt = hs = ls = he = le = 0
            for i_l in xrange(K):
                if phi[i_l, i_v] - avg > -1e-8:
                    high[he] = phi[i_l, i_v]
                    high_map[he] = i_l
                    he += 1
                else:
                    low[le] = phi[i_l, i_v]
                    low_map[le] = i_l
                    le += 1

            while ls != le:

                table_h[i_v, table_cnt] = high_map[hs]
                table_l[i_v, table_cnt] = low_map[ls]
                table_p[i_v, table_cnt] = low[ls]

                table_cnt += 1
                after = high[hs] + low[ls] - avg
                ls += 1
                high[hs] = after

                while (high[hs] - avg <= -1e-8) and (hs < he-1):

                    table_l[i_v, table_cnt] = high_map[hs]
                    table_p[i_v, table_cnt] = high[hs]
                    hs += 1
                    table_h[i_v, table_cnt] = high_map[hs]
                    table_cnt += 1
                    high[hs] += high[hs-1] - avg

            while hs != he:
                table_h[i_v, table_cnt] = table_l[i_v, table_cnt] = high_map[hs]
                table_p[i_v, table_cnt] = high[hs]
                hs += 1
                table_cnt += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.overflowcheck(True)
cdef int rand_k(int K):
    cdef int r
    cdef int buckets = int(RAND_MAX) / K
    cdef int limit = buckets * K

    r = rand()
    while r >= limit:
        r = rand()

    r /= buckets
    return r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.overflowcheck(True)
cdef int rand_k_inner(int K, int rand_max):
    cdef int r
    cdef int buckets = rand_max / K
    cdef int limit = buckets * K

    r = rand()
    while r >= limit:
        r = rand()

    r /= buckets
    return r

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double rand_u():
    r = (<double> rand() / (RAND_MAX))
    while 1 - r < 1e-8:
        r = (<double> rand() / (RAND_MAX))
    return r





def set_srand():
    srand(time(NULL))


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_ids(np.ndarray[dtype_t, ndim=2] Adk_avg,
                 np.ndarray[dtype_t, ndim=2] Bkw_avg,
                 np.ndarray[uitype_t, ndim=2] Adk,
                 np.ndarray[uitype_t, ndim=2] Bkw,
                 np.ndarray[dtype_t, ndim=2] phi,
                 np.ndarray[dtype_t, ndim=1] uni_rvs,
                 np.ndarray[np.uint8_t, ndim=1, cast=True] mask,
                 list doc_dicts,
                 list z,
                 double alpha,
                 int num_sim,
                 int burn_in):

    if not phi.flags.f_contiguous: phi = phi.copy('F')
    if not Adk.flags.c_contiguous: phi = phi.copy('C')
    #if not Bkw.flags.f_contiguous: phi = phi.copy('F')

    cdef Py_ssize_t D = Adk.shape[0]
    cdef Py_ssize_t K = Adk.shape[1]
    cdef Py_ssize_t W = Bkw.shape[1]
    cdef Py_ssize_t d, w, i, k, sim, word_cnt, zInit, zOld, zNew
    cdef Py_ssize_t rc_start = 0, rc_mid, rc_stop = K
    cdef double prob_sum, uni_rv
    cdef Py_ssize_t uni_idx = 0
    cdef np.ndarray[dtype_t, ndim=1] probs = np.zeros(K)
    cdef np.ndarray[dtype_t, ndim=1] cumprobs = np.linspace(0,1,K+1)[0:K]

    Adk.fill(0)
    Bkw.fill(0)
    Adk_avg.fill(0)
    Bkw_avg.fill(0)

    for d in range(D):
        doc_len = len(doc_dicts[d])
        z[d] = np.zeros(doc_len, dtype=np.int32)
        for i in xrange(doc_len):
            w = doc_dicts[d][i]
            if not mask[w]:
                continue

            zInit = rand_k(K)
            Adk[d,zInit] += 1
            Bkw[zInit,w] += 1
            z[d][i] = zInit

    for sim in xrange(num_sim):
        for d in xrange(D):

            doc_len = len(doc_dicts[d])
            for i in xrange(doc_len):
                zOld = z[d][i]
                w = doc_dicts[d][i]
                if not mask[w]:
                    continue

                prob_sum = 0
                for k in range(K):
                    cumprobs[k] = prob_sum
                    prob_sum +=  (alpha + Adk[d,k] - (k == zOld)) * phi[k, w]
                uni_rv = prob_sum  * uni_rvs[uni_idx]
                uni_idx += 1

                rc_start = 0
                rc_stop  = K
                while rc_start < rc_stop - 1:
                    rc_mid = (rc_start + rc_stop) // 2
                    if cumprobs[rc_mid] <= uni_rv:
                        rc_start = rc_mid
                    else:
                        rc_stop = rc_mid
                zNew = rc_start

                z[d][i] = zNew
                Adk[d,zOld]     -= 1
                Adk[d,zNew]     += 1
                Bkw[zOld,w]     -= 1
                Bkw[zNew,w]     += 1


        if sim >= burn_in:
            Adk_avg += Adk
            Bkw_avg += Bkw

    Adk_avg /= (num_sim - burn_in)
    Bkw_avg /= (num_sim - burn_in)





@cython.boundscheck(False)
@cython.wraparound(False)
def sample_z_id_o(np.ndarray[dtype_t, ndim=2] Adk_avg,
                 np.ndarray[dtype_t, ndim=2] Bkw_avg,
                 np.ndarray[uitype_t, ndim=2] Adk,
                 np.ndarray[uitype_t, ndim=2] Bkw,
                 np.ndarray[dtype_t, ndim=2] phi,
                 np.ndarray[dtype_t, ndim=1] uni_rvs,
                 list doc_dicts,
                 list z,
                 double alpha,
                 int num_sim,
                 int burn_in):

    if not phi.flags.f_contiguous: phi = phi.copy('F')
    if not Adk.flags.c_contiguous: phi = phi.copy('C')
    cdef Py_ssize_t D = Adk.shape[0]
    cdef Py_ssize_t K = Adk.shape[1]
    cdef Py_ssize_t W = Bkw.shape[1]
    cdef Py_ssize_t d, w, i, k, sim, word_cnt, zInit, zOld, zNew, doc_len
    cdef Py_ssize_t rc_start = 0, rc_mid, rc_stop = K
    cdef double prob_sum, uni_rv
    cdef Py_ssize_t uni_idx = 0
    cdef np.ndarray[dtype_t, ndim=1] probs = np.zeros(K)
    cdef np.ndarray[dtype_t, ndim=1] cumprobs = np.linspace(0,1,K+1)[0:K]


    Adk.fill(0)
    Bkw.fill(0)
    Adk_avg.fill(0)
    Bkw_avg.fill(0)

    for d in range(D):
        doc_len = len(doc_dicts[d])
        z[d] = np.zeros(doc_len, dtype=np.int32)
        for i in xrange(doc_len):
            w = doc_dicts[d][i]
            zInit = rand_k(K)
            Adk[d,zInit] += 1
            Bkw[zInit,w] += 1
            z[d][i] = zInit

    for sim in xrange(num_sim):
        for d in xrange(D):

            doc_len = len(doc_dicts[d])
            for i in xrange(doc_len):
                zOld = z[d][i]
                w = doc_dicts[d][i]

                prob_sum = 0
                for k in range(K):
                    cumprobs[k] = prob_sum
                    prob_sum +=  (alpha + Adk[d,k] - (k == zOld)) * phi[k, w]
                uni_rv = prob_sum  * uni_rvs[uni_idx]
                uni_idx += 1

                rc_start = 0
                rc_stop  = K
                while rc_start < rc_stop - 1:
                    rc_mid = (rc_start + rc_stop) // 2
                    if cumprobs[rc_mid] <= uni_rv:
                        rc_start = rc_mid
                    else:
                        rc_stop = rc_mid
                zNew = rc_start

                z[d][i] = zNew
                Adk[d,zOld]     -= 1
                Adk[d,zNew]     += 1
                Bkw[zOld,w]     -= 1
                Bkw[zNew,w]     += 1


        if sim >= burn_in:
            Adk_avg += Adk
            Bkw_avg += Bkw

    Adk_avg /= (num_sim - burn_in)
    Bkw_avg /= (num_sim - burn_in)


def gen_obj(int V):

    table_set = Samples.__new__(Samples, V)
    return table_set

def kill_obj(int V, Samples stale):
    cdef int i
    free(stale.w_pointer)
    for i in xrange(V):
        free(stale.w_sample[i])
    free(stale.w_sample)

cdef class Samples:
    cdef int** w_sample
    cdef int* w_pointer

    def __cinit__(self, int V):
        self.w_sample = <int **>malloc(V*sizeof(int*))
        self.w_pointer = <int *>malloc(V*sizeof(int))
        cdef int i
        for i in xrange(V):
            self.w_pointer[i] = 0
