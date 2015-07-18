from os import listdir
from pickle import dump, load
import matplotlib.pyplot as plt


def first(out_dir):
    good_out = []
    cnt = 0

    for filename in listdir(out_dir):

        f = open(out_dir + filename)

        prplx = []
        flag = False
        for line in f:
            nums = line.split()
            if float(nums[0]) < 3100: flag = True
            prplx.append(float(nums[0]))

        if flag:
            good_out += [(filename, prplx)]
            cnt += 1

    print cnt
    dump(good_out, open(out_dir+'parse_out', 'w'))


def second(out_dir):
    out = load(open(out_dir+'parse_out', 'r'))
    # fig, ax = plt.subplots(4, 5)
    sty = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h']
    i = 0
    for n in xrange(19):
        plt.plot([x for x in xrange(len(out[n][1]))], out[n][1], sty[n], label=out[n][0][:11])
        plt.legend()
    # for k in xrange(4):
    #     for j in xrange(5):
    #
    #         try:
    #             ax[k][j].plot([x for x in xrange(len(out[i][1]))], out[i][1], label=out[i][0][:11])
    #             i += 1
    #             ax[k][j].legend()
    #         except:
    #             for n in xrange(19):
    #                 ax[k][j].plot([x for x in xrange(len(out[n][1]))], out[n][1], sty[n], label=out[n][0][:11])
    #                 ax[k][j].legend()
    plt.show()



if __name__ == '__main__':
    out_dir = 'F://Work//MCMC//experiments//s_out_2//'
    # first(out_dir)
    second(out_dir)
