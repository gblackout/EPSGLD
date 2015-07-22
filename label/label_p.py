import numpy as np


def get_code(arr):
    """
    get the hash code of an array, we parse by row
    """
    return (arr - np.mean(arr, axis=1)[:, np.newaxis]) > 0


def compare(c1, c2):
    """
    compare the distance of c1 and c2, input vector,
    """
    # TODO probably consider the miss match of 1
    return 2*np.sum(np.logical_and(c1, c2)) - max(c1.sum(), c2.sum())


def search(c1, c2):
    """
    return the indices that map c2 to c1, input vector or matrix
    """
    lenn = c2.shape[0]
    map_list = np.zeros(lenn, dtype=np.int32)
    wait_list = range(c1.shape[0])

    for i in xrange(lenn):
        score = 0
        ind = 0

        for j in xrange(len(wait_list)):
            tmp = compare(c1[wait_list[j]], c2[i])
            if tmp > score:
                score = tmp
                ind = j
        map_list[wait_list.pop(ind)] = i

    return map_list


# a = np.random.dirichlet([0.0001 for _ in xrange(1000)], size=1000)
# b = np.random.permutation(a)
#
# map_list = search(get_code(a), get_code(b))
# b = b[map_list]
#
# cnt = 0
# for i in xrange(a.shape[0]):
#     if ((a[i, :] - b[i, :]) > 10**(-8)).any():
#         cnt += 1
# print cnt

