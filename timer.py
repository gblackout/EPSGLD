from time import time


class Timer(object):

    def __init__(self, cnt=0, go=False):
        self.cnt = float(cnt)
        if go:
            self.start = time()
            self.is_cnting = True
        else:
            self.start = None
            self.is_cnting = False

    def go(self):
        assert ~self.is_cnting, 'cannot go() twice'
        self.start = time()
        self.is_cnting = True

    def stop(self):
        assert self.is_cnting, 'does not start yet'
        self.cnt += time() - self.start
        self.is_cnting = False
        return self

    def __call__(self):
        return self.cnt