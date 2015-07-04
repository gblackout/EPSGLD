__author__ = 'YY'

import numpy as np
import time


if __name__ == '__main__':
    a = np.ones((1, int(1e5)), dtype=np.int32)
    b = np.ones((int(1e5), 1), dtype=np.int32)

    start = time.time()
    np.log(np.dot(a, b))
    print time.time() - start

