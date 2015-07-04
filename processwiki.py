# functions to convert wikipedia articles saved as
# (article_as_string, title)
# to
# (title, ({wordid : trainwordct}, {wordid: testwordct}))
# and save them back to disk

from __future__ import division

from collections import Counter, deque
from itertools import islice
from wikirandom import get_random_wikipedia_article

import cPickle as pickle
import re
import copy

def online_wiki_docs():
    # This is slow, for illustration only
    while True:
        yield get_random_wikipedia_article()


def load_list(pkl_file, num_items=3300000):
    with open(pkl_file,'rb') as f:
        while num_items > 0:
            try:
                yield pickle.load(f)
                num_items -= 1
            except EOFError:
                raise StopIteration


def clean_doc(doc, vocab):
    """ given doc and user-defined vocab return list like ['you','me'] where the word can be found in vocab"""

    doc = doc.lower()
    doc = re.sub(r'-', ' ', doc)
    doc = re.sub(r'[^a-z ]', '', doc)
    doc = re.sub(r' +', ' ', doc)
    return [word for word in doc.split() if word in vocab]


def parse_docs(name_docs, vocab):
    """ generate a list of tuples like [ ('my novel',{...},{...}) , ('my novel',{...},{...}) ]"""
    for doc, name in name_docs:
        words = clean_doc(doc, vocab)

        # Hold back every 10th word to get an online estimate of perplexity
        # thus train_words takes the same form as words
        train_words = [vocab[w] for (i, w) in enumerate(words) if i % 10 != 0]
        test_words = [vocab[w] for (i, w) in enumerate(words) if i % 10 == 0]

        # for a list ['1','1','2','3','4']
        # Counter returns the dict like this {'1':2,'2':1,'3':1,'4':1}
        # here the dict takes the form like this {3846:1,2393:1,8839:2} since the word is mapped into the vocab
        train_cntr = Counter(train_words)
        test_cntr = Counter(test_words)
        yield (name, train_cntr, test_cntr)



def take_every(n, iterable):
    """ returns the list like [ [('',{},{}),...] , [] ], the 2nd dimension list has n tuples,
    thus each of them is a mini-batch"""
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def group_up(n, N, iterable):
    i = iter(iterable)
    ctr = 0
    while ctr < N:
        piece = list(islice(i,n))
        ctr += n
        yield piece


def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def save_data(source, out_file):
    with open(out_file,'wb') as out:
        for item in source:
            pickle.dump(item, out, protocol=-1)

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


def create_vocab(vocab_file):
    dictnostops = open(vocab_file).readlines()
    vocab = dict()

    # the dictnostops is a list like ['you','me']
    # the enumerate returns a object containing the tuple like (1,'you') (2,'me')
    # syntax like list(enumerate(a)) will create the list like [(1,'you'),(2,'me')]
    for idx,word in enumerate(dictnostops):
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        vocab[word] = idx
    return vocab
