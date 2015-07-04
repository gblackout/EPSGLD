from __future__ import division
import sys
import urllib2
import re
import threading
from collections import Counter
from itertools import islice
import cPickle as pickle
import math
import copy
from os import listdir, path, rename, remove, mkdir
import processwiki as pw
from sys import argv


def get_and_save(num, vocab, index, num_per_file=None, num_of_files=None):
    """ randomly get the online wiki docs and save into files, you can specify the num of docs in each file or
    how many files you want to save, when specified one another one should left blank"""
    if num_per_file is not None and num_of_files is not None:
        print 'param error'
        return
    else:
        if num_of_files is not None:
            num_per_file = math.ceil(num / num_of_files)
        while num-num_per_file > 0:
            data = parse_docs(get_random_wikipedia_articles(num_per_file), vocab)
            save_data(data, file_index=index)
            index += 1
            num -= num_per_file
            print 'docs left: %i' % num
            sys.stdout.flush()
        if num != 0:
            data = parse_docs(get_random_wikipedia_articles(num), vocab)
            save_data(data, file_index=index)



def get_and_save_dump(ddir, vocab, index, num_per_file=10000):
    """ randomly get the online wiki docs and save into files, you can specify the num of docs in each file or
    how many files you want to save, when specified one another one should left blank"""

    file_list = listdir(ddir)
    data = []
    cnt = 0
    for fn in file_list:
        print 'main--->parsing file %i of %i' % (cnt, len(file_list))
        cnt += 1
        f = open(ddir + '\\' + fn)
        if len(data) < num_per_file:
            data += parse_docs_dump(f, vocab)
        else:
            t = int(len(data) / num_per_file)
            for i in range(t):
                save_data(data[i*num_per_file:(i+1)*num_per_file], file_index=index)
                index += 1
            data = data[t*num_per_file:] + list(parse_docs_dump(f, vocab))

    if len(data) > 0:
        save_data(data, file_index=index)


def get_random_wikipedia_articles(n):
    """ Downloads n articles in 8 threads """
    maxthreads = 8
    WikiThread.articles = list()
    wtlist = list()
    for i in range(0, n, maxthreads):
        print 'get %i docs' % i
        for j in range(i, min(i+maxthreads, n)):
            wtlist.append(WikiThread())
            wtlist[len(wtlist)-1].start()
        for j in range(i, min(i+maxthreads, n)):
            wtlist[j].join()
    return WikiThread.articles


def get_random_wikipedia_article():
    """Downloads a randomly selected Wikipedia article via http://en.wikipedia.org/wiki/Special:Random) """
    failed = True
    timeout = 1
    while failed:
        articletitle = None
        failed = False
        try:
            req = urllib2.Request('http://en.wikipedia.org/wiki/Special:Random',
                                  None, { 'User-Agent' : 'x'})
            f = urllib2.urlopen(req, timeout=3)
            while not articletitle:
                line = f.readline()
                result = re.search(r'title="Edit this page" href="/w/index.php\?title=(.*)\&amp;action=edit" /\>', line)
                if (result):
                    articletitle = result.group(1)
                    break
                elif (len(line) < 1):
                    sys.exit(1)
            req = urllib2.Request('http://en.wikipedia.org/w/index.php?title=Special:Export/%s&action=submit' \
                                      % (articletitle),
                                  None, { 'User-Agent' : 'x'})
            f = urllib2.urlopen(req, timeout=timeout)
            all = f.read()
            print '--->D'
            sys.stdout.flush()
        except:  # (urllib2.HTTPError, urllib2.URLError):
            # timeout *= 2
            print 'oops. there was a failure downloading %s. retrying...' \
                % articletitle
            sys.stdout.flush()
            failed = True
            continue

        try:
            all = re.search(r'<text.*?>(.*)</text', all, flags=re.DOTALL).group(1)
            all = re.sub(r'\n', ' ', all)
            all = re.sub(r'\{\{.*?\}\}', r'', all)
            all = re.sub(r'\[\[Category:.*', '', all)
            all = re.sub(r'==\s*[Ss]ource\s*==.*', '', all)
            all = re.sub(r'==\s*[Rr]eferences\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks and [Rr]eferences==\s*', '', all)
            all = re.sub(r'==\s*[Ss]ee [Aa]lso\s*==.*', '', all)
            all = re.sub(r'http://[^\s]*', '', all)
            all = re.sub(r'\[\[Image:.*?\]\]', '', all)
            all = re.sub(r'Image:.*?\|', '', all)
            all = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', all)
            all = re.sub(r'\&lt;.*?&gt;', '', all)
        except:
            # Something went wrong, try again. (This is bad coding practice.)
            print 'oops. there was a failure parsing %s. retrying...' \
                % articletitle
            sys.stdout.flush()
            failed = True
            continue
    print '--->E'
    return (all, articletitle)


class WikiThread(threading.Thread):
    articles = list()
    lock = threading.Lock()

    def run(self):
        article = get_random_wikipedia_article()
        WikiThread.lock.acquire()
        WikiThread.articles.append(article)
        WikiThread.lock.release()


def load_list(pkl_file, num_items=3300000):
    """And load like this data = pw.load_list('stored.pkl')
       output is object of generator"""
    if path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            while num_items > 0:
                try:
                    yield pickle.load(f)
                    num_items -= 1
                except EOFError:
                    raise StopIteration
    else:
        print '*** file %s not exist ***' % pkl_file

def trim_set(folder_path):
    file_list = listdir(folder_path)
    for file in file_list:
        docs = list(load_list(folder_path+'/'+file))
        for doc in docs:
            if len(doc) == 2:
                doc += [{1:1}]
        save_data()





def parse_docs(name_docs, vocab):
    """ input: a list [('you are a pig','title'), (...)]
        output: a list of tuples like [ ('my novel',{...},{...}) , ('my novel',{...},{...}) ]
        i modify it into the form [ ('my novel',{...}) , ('my novel',{...}) ]"""
    for doc, name in name_docs:
        words = clean_doc(doc, vocab)

        # Hold back every 10th word to get an online estimate of perplexity
        # thus train_words takes the same form as words
        # train_words = [vocab[w] for (i, w) in enumerate(words) if i % 10 != 0]
        # test_words = [vocab[w] for (i, w) in enumerate(words) if i % 10 == 0]

        train_words = [vocab[w] for (i, w) in enumerate(words)]

        # for a list ['1','1','2','3','4']
        # Counter returns the dict like this {'1':2,'2':1,'3':1,'4':1}
        # here the dict takes the form like this {3846:1,2393:1,8839:2} since the word is mapped into the vocab
        train_cntr = Counter(train_words)
        yield (name, train_cntr)

def parse_docs_dump(file, vocab):
    """ input: a file extracted by the WikiExtractor
        output: a list of tuples like [ ('my novel',{...},{...}) , ('my novel',{...},{...}) ]
        i modify it into the form [ ('my novel',{...}) , ('my novel',{...}) ]"""
    name_docs = []
    tmp = ''
    flag = False
    cnt = 0
    for line in file:
        if '<doc id' in line:
            flag = True
            continue
        if r'</doc>' in line:
            flag = False
            name_docs += [(remove_comma(tmp), 'pesudonym')]
            tmp = ''
        if flag:
            tmp += line

    for doc, name in name_docs:
        words = clean_doc(doc, vocab)

        train_words = [vocab[w] for (i, w) in enumerate(words)]

        train_cntr = Counter(train_words)
        yield (name, train_cntr)

def remove_comma(all):
    all = re.sub(r'\n', ' ', all)
    all = re.sub(r'\{\{.*?\}\}', r' ', all)
    all = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', all)
    all = re.sub(r'\&lt;.*?&gt;', ' ', all)
    return all

def remove_comma_outter(all):
    all = re.sub(r'\n', ' ', all)
    all = re.sub(r'\{\{.*?\}\}', r' ', all)
    all = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', all)
    all = re.sub(r'\&lt;.*?&gt;', ' ', all)

    all = all.lower()
    all = re.sub(r'-', ' ', all)
    all = re.sub(r'[^a-z ]', '', all)
    all = re.sub(r' +', ' ', all)

    return all.split()

def clean_doc(doc, vocab):
    """ given doc and user-defined vocab return list like ['you','me'] where the word can be found in vocab"""

    doc = doc.lower()
    doc = re.sub(r'-', ' ', doc)
    doc = re.sub(r'[^a-z ]', '', doc)
    doc = re.sub(r' +', ' ', doc)
    return [word for word in doc.split() if word in vocab]

def take_every(n, iterable):
    """ returns the list like [ [('',{},{}),...] , [] ], the 2nd dimension list has n tuples,
    thus each of them is a mini-batch"""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def save_data(source, out_file=None, file_index=None, dir='./'):
    """Can save data to a file like this processwiki.save_data(itertools.islice(data, 1000), 'stored.pkl')
        you can specify the place of the file your own or you must specify the naming index of the file"""
    if out_file is None:
        fp = dir + 'stored%i.pkl' % file_index
        with open(fp, 'wb') as out:
            for item in source:
                pickle.dump(item, out, protocol=-1)
    elif out_file is not None:
        with open(dir + out_file, 'wb') as out:
            for item in source:
                pickle.dump(item, out, protocol=-1)
    else:
        print 'write error'


def create_vocab(vocab_file):
    dictnostops = open(vocab_file).readlines()
    vocab = dict()

    # the dictnostops is a list like ['you','me']
    # the enumerate returns a object containing the tuple like (1,'you') (2,'me')
    # syntax like list(enumerate(a)) will create the list like [(1,'you'),(2,'me')]
    for idx, word in enumerate(dictnostops):
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        vocab[word] = idx
    return vocab

def get_test_set(docs):
    """input: [ ('novel',{1234:1,4567:1}) , (...) ]
       output: [ ('novel',{},{}) ,(...) ] that is extract a test set for the heldout set"""
    cnt = 0
    out_docs = []
    for name, doc in docs:
        test = {}
        doc_c = copy.deepcopy(doc)
        for word in doc:
            if cnt % 10 == 0:
                test[word] = doc[word]
                del doc_c[word]
            cnt += 1
        out_docs += [(name, doc_c, test)]
    return out_docs

def test_full_set():
    V = len(pw.create_vocab('wiki.vocab'))
    for i in xrange(20, 21):
        a = list(load_list('./corpus/stored%i.pkl' % i))
        try:
            (names, train_cts) = zip(*a)
        except:
            (names, train_cts, test_cts) = zip(*a)
        train_cts = list(train_cts)

        num_words = []
        for cnt in train_cts:
            num_words += [sum(cnt.values())]

        # num_words = [sum(cnt.values) for cnt in train_cts]
        print sum(num_words[:])

        # if len(train_cts) != 10000:
        #     print i


def test_V(down, up):
    V = [i for i in xrange(7702)]
    for i in xrange(down, up):
        a = list(load_list('./corpus/stored%i.pkl' % i))
        try:
            (names, train_cts) = zip(*a)
        except:
            (names, train_cts, test_cts) = zip(*a)
        train_cts = list(train_cts)

        for doc in train_cts:
            for w in doc.keys():
                try:
                    V.remove(w)
                except:
                    pass

        print 'at cts %i remain %i' % (i, len(V))

def reform(down, up):
    for i in xrange(down, up):
        a = list(load_list('./small_test/stored%i.pkl' % i))
        try:
            (names, train_cts) = zip(*a)
        except:
            (names, train_cts, test_cts) = zip(*a)
        train_cts = list(train_cts)

        out = []
        for doc in train_cts:
            doc_list = [w for (w, cntr) in doc.iteritems() for _ in xrange(cntr)]
            out += [doc_list]

        save_data(out, out_file='./small_test/stored%i_simple.pkl' % i)

def reform_test(down, up):
    for i in xrange(down, up):
        a = list(load_list('./small_test/stored_test%i.pkl' % i))
        (names, train_cts, test_cts) = zip(*a)
        train_cts = list(train_cts)
        test_cts = list(test_cts)

        num_doc = len(train_cts)

        out = []

        for d in xrange(num_doc):
            doc_list = [w for (w, cntr) in train_cts[d].iteritems() for _ in xrange(cntr)]
            doc_list_test = [w for (w, cntr) in test_cts[d].iteritems() for _ in xrange(cntr)]
            out += [(doc_list, doc_list_test)]

        save_data(out, out_file='./small_test/stored_test%i_simple.pkl' % i)

def con_file(input, index=0, indent=0):
    """ input should end with / """
    buf = []
    for file_name in listdir(input):
        if 'incomplete' in file_name:
            print 'processing:', file_name
            buf += pickle.load(open(input+file_name, 'r'))
            if len(buf) > 10000:
                pickle.dump(buf[:10000], open('con_saved_'+str(indent)+"_"+str(index), 'w'))
                buf = buf[10000:]
                index += 1

    if len(buf) != 0:
        pickle.dump(buf, open('con_saved_' + str(indent) + "_" + str(index)+'incomplete', 'w'))

from operator import itemgetter
def get_words(input):
    """ need wiki.vocab in same directory with code not the corpus"""
    w_map = {}
    for file_name in listdir(input):
        print '*************' + file_name + '*************'
        set = pickle.load(open(input + file_name, 'r'))
        cnt = 0
        for d in set:
            if cnt % 500 == 0: print 'doc:', cnt
            for w in d:
                if w in w_map:
                    w_map[w] += 1
                else:
                    w_map[w] = 1
            cnt += 1

    pickle.dump(w_map, open('raw_map_saved', 'w'))
    vocab = create_vocab('wiki.vocab')

    print 'find max_f'
    max_f = max([(v, w_map[v]) for v in vocab if v in w_map], key=itemgetter[1])
    print 'found:', max_f

    print 'exclude and sort'
    w_list = [(k, v) for (k, v) in w_map.iteritems() if v <= max_f[1]]
    w_list.sort(key=lambda tup: tup[1], reverse=True)

    print 'gen w_map'
    if len(w_list) < int(1e5):
        print 'not enough'
        print max_f
    else:
        w_map = {}
        for i in xrange(int(1e5)):
            if i % 1000 == 0:
                print i
            w_map[w_list[i][0]] = i

        pickle.dump(w_map, open('w_map_saved', 'w'))

def get_w_half():

    w_map = pickle.load(open('raw_map_saved', 'r'))
    vocab = create_vocab('wiki.vocab')
    stop_set = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost",
                "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst",
                "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
                "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been",
                "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill",
                "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
                "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either",
                "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
                "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five",
                "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get",
                "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby",
                "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie",
                "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
                "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine",
                "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely",
                "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not",
                "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other",
                "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
                "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious",
                "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow",
                "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten",
                "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter",
                "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this",
                "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top",
                "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very",
                "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where",
                "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
                "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without",
                "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"}
    print 'find max_f'
    w_list = []
    for v in w_map:
        if (not v in stop_set) and len(v) > 2:
            w_list += [(v, w_map[v])]

    w_list.sort(key=lambda tup: tup[1], reverse=True)
    pickle.dump(w_list[:100000], open('freq_list', 'w'))

    # print 'gen w_map'
    # if len(w_list) < int(1e5):
    #     print 'not enough'
    #     # print max_f
    # else:
    #     w_map = {}
    #     for i in xrange(int(1e5)):
    #         if i % 1000 == 0:
    #             print i
    #         w_map[w_list[i][0]] = i
    #
    #     pickle.dump(w_map, open('w_map_saved', 'w'))

import numpy as np
def raw_token(input, output, w_map, offset):
    cnt = 0
    w_map = pickle.load(open(w_map, 'r'))
    for file_name in listdir(input):
        print 'set: ', file_name, ' cnt: ', cnt
        set = pickle.load(open(input + file_name, 'r'))
        t_set = [[w_map[v] for v in d if v in w_map] for d in set]
        np.save(output + 't_saved_' + str(cnt+int(offset)*44), t_set)
        # pickle.dump(t_set, open(output + 't_saved_' + str(cnt+int(offset)*44), 'w'))
        cnt += 1


def batchize_doc(input, output, W, batch_size):
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        print 'set: ', file_name

        b_set = []
        batch_doc = []
        batch_map = {}
        batch_mask = np.zeros(W, dtype=bool)
        cnt = 0

        for d in set:

            if cnt % 2000 == 0: print 'doc: ', cnt

            batch_doc += [d]
            for w in d:
                if w in batch_map:
                    batch_map[w] += 1
                else:
                    batch_map[w] = 1
                    batch_mask[w] = 1
            cnt += 1
            if cnt % batch_size == 0 and cnt != 0:
                b_set += [(batch_doc, batch_map, batch_mask.copy())]
                batch_doc = []
                batch_map = {}
                batch_mask.fill(0)

        pickle.dump(b_set, open(output + 'b_' + file_name[3:], 'w'))


def batchize_doc_4(input, output, W, batch_size):
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        print 'set: ', file_name

        b_set = []
        b_subset = []
        batch_doc = []
        batch_map = {}
        batch_mask = np.zeros(W, dtype=bool)
        cnt = 0

        for d in set:

            if cnt % 2000 == 0: print 'doc: ', cnt

            batch_doc += [d]
            for w in d:
                if w in batch_map:
                    batch_map[w] += 1
                else:
                    batch_map[w] = 1
                    batch_mask[w] = 1
            cnt += 1
            if cnt % batch_size == 0 and cnt != 0:
                b_subset += [batch_doc]
                batch_doc = []
                if len(b_subset) == 4:
                    b_set += [(b_subset, batch_map, batch_mask.copy(), True)]
                    batch_map = {}
                    batch_mask.fill(0)
                    b_subset = []

        pickle.dump(b_set, open(output + 'b_' + file_name[3:], 'w'))

def validate_set(input):
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))
        lenn = len(set)
        if lenn != 10000:
            print '******', file_name, lenn
        else:
            print file_name

def doc_anay(input, ident):

    batch_w_cnt = []
    w_samples = []

    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        print 'set: ', file_name

        for batch in set:
            batch_w_cnt += [batch[2].sum()]
            sort_samples = sorted(batch[1].values(), reverse=True)
            for i in xrange(min(10, len(sort_samples))):
                w_samples += [sort_samples[i]]

    pickle.dump(batch_w_cnt, open('batch_w_cnt_'+str(ident), 'w'))
    pickle.dump(w_samples, open('w_samples_' + str(ident), 'w'))

def mk_sta():

    f_1 = open('w_samples', 'w')
    f_2 = open('batch_w_cnt', 'w')

    for i in xrange(1, 11):

        print 'rec: ', i

        w_samples = pickle.load(open('w_samples_' + str(i), 'r'))
        batch_w_cnt = pickle.load(open('batch_w_cnt_' + str(i), 'r'))

        for e in w_samples:
            f_1.write('%i\n' % e)
        for e in batch_w_cnt:
            f_2.write('%i\n' % e)

    f_1.close()
    f_2.close()


def set_flag(input, max_sample, max_w):
    cnt = 0
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        print 'set: ', file_name

        for i in xrange(len(set)):
            if (max(set[i][1].values()) > max_sample) or (set[i][2].sum() > max_w):
                set[i] = (set[i][0], set[i][1], set[i][2], False)
                cnt += 1

        pickle.dump(set, open(input + file_name, 'w'))

    print 'excluded cnt: ', cnt

def batch_2_file(input, output):
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))
        cnt = 0

        print 'set: ', file_name

        for subset in set:
            if subset[3] == True:
                pickle.dump(subset, open(output + file_name + '_' + str(cnt), 'w'))
                cnt += 1

def test_for_sample(input):
    cnt = 0
    cnt_file = 0
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        if cnt_file % 500 == 0: print 'set cnt: ', cnt_file
        w_sample = set[1].values()
        if sum(w_sample) * set[2].sum() / float(len(w_sample)) > 1.5625e7:
            cnt += 1
            print cnt
        cnt_file += 1

    print cnt

def reform_4batch(input, output, W):
    """ original form: [ [[d],[d],[d],[d]], map{}, mask[], flag ]
        objective form: [ [[d],[d],[d],[d]], map[], mask[], flag, [mask[d], mask[d], mask[d], mask[d]] ]"""
    cnt_file = 0
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        if cnt_file % 200 == 0: print 'set cnt: ', cnt_file
        map_list = np.zeros(W, dtype=np.int32)

        for (w, cnt) in set[1].items(): map_list[w] = cnt

        mask_set = []
        for i in xrange(4):
            mask = np.zeros(W, dtype=bool)
            batch = set[0][i]
            for d in batch:
                for w in d:
                    mask[w] = 1
            mask_set += [mask]

        pickle.dump([set[0], map_list, set[2], set[3], mask_set], open(output + file_name, 'w'))

        cnt_file += 1


def get_right_batch(input):
    cnt_file = 0
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        if cnt_file % 200 == 0: print 'set cnt: ', cnt_file
        set[1].fill(0)
        set[2].fill(0)
        for mask in set[4]:
            mask.fill(0)

        for b_cnt in xrange(len(set[0])):
            for d in set[0][b_cnt]:
                for w in d:
                    set[1][w] += 1
                    set[2][w] = 1
                    set[4][b_cnt][w] = 1

        pickle.dump(set, open(input + file_name, 'w'))


def reduce_test_doc(input):
    test = pickle.load(open(input, 'r'))

    mask = np.zeros(100000, dtype=bool)
    for d in test:
        for w in d:
            mask[w] = 1

    for i in xrange(mask.shape[0]):
        if i % 3 == 0:
            mask[i] = 0

    num = mask.sum()
    i = mask.shape[0] - 1
    tic_toc = True
    while num > 10000:

        while not mask[i]: i -= 1

        if tic_toc:
            mask[i] = 0
            num -= 1

        tic_toc = not tic_toc
        i -= 1

    mask.dump('./test_doc_mask')


def reform_test_doc(input, mask):
    test = pickle.load(open(input, 'r'))
    mask = np.load(mask)

    for i in xrange(len(test)):
        dd = d = test[i]
        for w in dd:
            if not mask[w]: test[i] = d = filter(lambda a: a != w, d)

    pickle.dump(test, open('./test_doc', 'w'))


def re_test(input):
    test = pickle.load(open(input, 'r'))

    mm = []
    for d in test:
        remove = [i for i in xrange(len(d)) if i % 10 == 0]
        small = [d[x] for x in remove]
        mm += [[small, [i for (j, i) in enumerate(d) if j not in remove]]]

    pickle.dump(mm, open('./test_final', 'w'))


def test_for_n(input):
    num = []
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))

        num += [set[1].sum()]

    print input, max(num), sum(num) / len(num)


def rere_test(input, W):
    test = pickle.load(open(input, 'r'))

    trains = []
    tests = []
    mask = np.zeros(W, dtype=bool)
    map = np.zeros(W, dtype=np.int32)

    for d in test:
        trains += [d[1]]
        tests += [d[0]]

        for w in d[0]:
            mask[w] = 1

        for w in d[1]:
            mask[w] = 1
            map[w] += 1

    pickle.dump([trains, tests, mask, map], open('test_complete', 'w'))

def rename_set(input):
    cnt = 0
    for file_name in listdir(input):
        rename(input+file_name, input + 'saved_%i' % cnt)
        cnt += 1

def small_reform_large(input, output, batch_size, W):
    """ assume the set size is the multiple of the batch_size"""
    cnt = 0
    input_test = input + './test/'
    input += './train/'
    for file_name in listdir(input):
        set = list(load_list(input + file_name))
        batch_set = []
        mask_set = []
        mask_g = np.zeros(W, dtype=bool)
        map_g = np.zeros(W, dtype=np.int32)

        while len(set) > 0:
            batch = set[:batch_size]
            set = set[batch_size:]
            mask = np.zeros(W, dtype=bool)
            for d in batch:
                for w in d:
                    mask[w] = 1
                    map_g[w] += 1

            batch_set += [batch]
            mask_set += [mask]
            mask_g += mask

        pickle.dump([batch_set, map_g, mask_g, True, mask_set], open(output + 'saved_%i' % cnt, 'w'))
        cnt += 1


    for file_name in listdir(input_test):
        (holdout_train_cts, holdout_test_cts) = zip(*list(load_list(input_test + file_name)))
        holdout_train_cts = list(holdout_train_cts)
        holdout_test_cts = list(holdout_test_cts)

        mask_g = np.zeros(W, dtype=bool)
        map_g = np.zeros(W, dtype=np.int32)

        for d in holdout_train_cts+holdout_test_cts:
            for w in d:
                mask_g[w] = 1
                map_g[w] += 1

        pickle.dump([holdout_train_cts, holdout_test_cts, mask_g, map_g], open(output + 'saved_test', 'w'))

def store_in_np(input):
    for file_name in listdir(input):
        set = pickle.load(open(input + file_name, 'r'))
        np.save(input+file_name, set)
        remove(input+file_name)


def change_size(input):
    cnt = 0
    folder = 1
    total = 432
    each = 19

    for i in xrange(1, 24):
        mkdir(input + 'a' + str(i))

    for i in xrange(1, 11):
        for file_name in listdir(input + str(i) + '/'):
            if 'test' not in file_name:
                print i, '----->', file_name
                rename(input + str(i) + '/' + file_name, input + 'a' + str(folder) + '/' + file_name + 'x')
                cnt += 1
                if cnt == each:
                    folder += 1
                    cnt = 0
                    if folder > 23: folder = 23

    for i in xrange(1, 24):
        cnt = 0
        for file_name in listdir(input + 'a' + str(i) + '/'):
            print i, '*****>', file_name
            rename(input + 'a' + str(i) + '/' + file_name, input + 'a' + str(i) + '/' + 't_saved_%i.npy' % cnt)
            cnt += 1

    # cnt = 0
    # for file_name in listdir('home/lijm/WORK/yuan/b4_ff/10/'):
    #     if 'test' not in file_name:
    #         if cnt != 420:
    #             rename('home/lijm/WORK/yuan/b4_ff/10/'+file_name, 'home/lijm/WORK/yuan/b4_ff/a22/'+file_name+'x')
    #             cnt += 1
    #         else:
    #             rename('home/lijm/WORK/yuan/b4_ff/10/' + file_name, 'home/lijm/WORK/yuan/b4_ff/a23/' + file_name + 'x')
    #

    #
    #     cnt = 0
    #     for file_name in listdir(input + 'a' + str(i) + '/'):
    #         print i, '*****>', file_name
    #         rename(input + 'a' + str(i) + '/' + file_name, input + 'a' + str(i) + '/' + 'saved_%i' % cnt)
    #         cnt += 1
    for i in xrange(24):
        cnt = 0
        while path.isfile(input + 'a' + str(i) + '/' + 't_saved_%i.npy' % cnt):
            cnt += 1
        print i, cnt, path.isfile(input + 'a' + str(i) + '/' + 'test_doc')


def cnt_token(input):
    cnt = 0
    for i in xrange(1, 11):
        print i
        for file_name in listdir(input + str(i) + '/'):
            if 'test' in file_name: continue
            set = np.load(input + str(i) + '/' + file_name)
            cnt += sum([len(set[x]) for x in xrange(10000)])

if __name__ == '__main__':
    # if len(argv) != 4:
    #     print 'try again'
    # else:
    #     con_file(argv[1], index=int(argv[2]), indent=int(argv[3]))

    # if len(argv) != 2:
    #     print 'try again'
    # else:
    #     get_words(argv[1])

    # get_w_half()

    # if len(argv) != 5:
    #     print 'try again'
    # else:
    #     raw_token(argv[1], argv[2], pickle.load(open(argv[3], 'r')), argv[4])

    # if len(argv) != 2:
    #     print 'try again'
    # else:
    #     validate_set(argv[1])

    # if len(argv) != 5:
    #     print 'try again'
    # else:
    #     batchize_doc(input=argv[1], output=argv[2], W=int(argv[3]), batch_size=int(argv[4]))

    # if len(argv) != 3:
    #     print 'try again'
    # else:
    #     doc_anay(input=argv[1], ident=int(argv[2]))

    # mk_sta()

    # if len(argv) != 5:
    #     print 'try again'
    # else:
    #     batchize_doc_4(input=argv[1], output=argv[2], W=int(argv[3]), batch_size=int(argv[4]))

    # if len(argv) != 4:
    #     print 'try again'
    # else:
    #     set_flag(input=argv[1], max_sample=int(argv[2]), max_w=int(argv[3]))

    # if len(argv) != 3:
    #     print 'try again'
    # else:
    #     batch_2_file(input=argv[1], output=argv[2])

    # if len(argv) != 2:
    #     print 'try again'
    # else:
    #     test_for_sample(input=argv[1])

    # if len(argv) != 4:
    #     print 'try again'
    # else:
    #     reform_4batch(input=argv[1], output=argv[2], W=int(argv[3]))

    # if len(argv) != 2:
    #     print 'try again'
    # else:
    #     reduce_test_doc(input=argv[1])

    if len(argv) != 2:
        print 'try again'
    else:
        # store_in_np(input=argv[1])
        # rename_set(input=argv[1])
        # change_size(input=argv[1])
        cnt_token(input=argv[1])


