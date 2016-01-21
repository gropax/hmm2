import os
from hmm.tagger import Tagger
from hmm.dependency_parser import ArcEagerParser, dp_features, train_dependency_parser, read_conll, ArcHybridParser


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')


def de_universal_corpus():
    train_path = os.path.join(DATA_DIR, 'de-universal-train.conll')
    dev_path = os.path.join(DATA_DIR, 'de-universal-dev.conll')
    test_path = os.path.join(DATA_DIR, 'de-universal-test.conll')

    train = read_conll(open(train_path, 'r', encoding='utf8'))
    dev = read_conll(open(dev_path, 'r', encoding='utf8'))
    test = read_conll(open(test_path, 'r', encoding='utf8'))

    return train, dev, test

def read_conll_tagging_data(infile, max_sent=None):
    """
    Lit un corpus au format conll et renvoie une liste de couples.
    Chaque couple contient deux listes de même longueur :
        la première contient les mots d'une phrase
        la seconde contient les tags correspondant
    Par exemple :

        [(["le", "chat", "dort"],["DET", "NOUN", "VERB"]),
          (["il", "mange", "."],["PRON", "VERB", "."]),
         ...etc
         ]

    """
    count = 0
    loc = infile.read()
    infile.close()
    sentences = []
    for sent_str in loc.strip().split('\n\n'):

        if max_sent is not None and max_sent < count:
            break

        count += 1
        lines = [line.split() for line in sent_str.split('\n')]
        words = []
        tags = []
        for _, word, _, coarse_pos, _, _, _, _, _, _ in lines:
            words.append(word)
            tags.append(coarse_pos)
        sentences.append((words,tags))
    return sentences


def read_map(file):
    map = {}
    for line in file:
        stripped = line.strip()
        if stripped:
            src, dst = stripped.split('\t')
            map[src] = dst
    return map

def map_corpus(corpus, map):
    for words, tags in corpus:
        yield (words, [map[t] for t in tags])
