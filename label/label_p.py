import numpy as np


print np.math.factorial(100)


# a = np.random.gamma(1, 1, (3, 3))
# b = np.random.permutation(a)
#
# norm_const_a = np.sum(a, 1)[:, np.newaxis]
# norm_const_b = np.sum(b, 1)[:, np.newaxis]
# a /= norm_const_a
# b /= norm_const_b
# print a
# print norm_const_a
#
# print norm_const_b
#
# for i in xrange(b.shape[0]):
#
#     ind = 0
#     pdt = 0
#
#     for j in xrange(a.shape[0]):
#
#         t_pdt = np.dot(a[j, :]/norm_const_a[j, 0], b[i, :]/norm_const_b[j, 0])
#
#         if t_pdt < pdt:
#             ind = j
#             pdt = t_pdt
#
#     print i, ind
