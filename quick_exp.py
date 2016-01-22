# encoding: utf8
from __future__ import division, print_function, unicode_literals, with_statement
from dependency_parser import *
from codecs import open as open
import random

def get_universal_treebank_data(language) :
    """
    Fonction pour lire les corpus d'entraînement, de développement et de test pour la langue language
    ("fr" pour français, "de" pour allemand, etc -> voir les données)
    """
    trainset ="./universal_treebanks_v2.0/std/{0}/{0}-universal-train.conll".format(language)
    devset   ="./universal_treebanks_v2.0/std/{0}/{0}-universal-dev.conll".format(language)
    testset  ="./universal_treebanks_v2.0/std/{0}/{0}-universal-test.conll".format(language)
    train_data = read_conll(open(trainset, "r", encoding = "utf8"))
    dev_data   = read_conll(open(devset, "r", encoding = "utf8"))
    test_data  = read_conll(open(testset, "r", encoding = "utf8"))
    return train_data, dev_data, test_data

def get_tiger_corpus_data() :
    """
    Fonction pour lire les corpus d'entraînement, de développement et de
    test du tiger corpus (voir lien sur didel).
    """
    trainset = "./data/german/tiger/train/german_tiger_train.conll"
    testset  = "./data/german/tiger/test/german_tiger_test.conll"
    train_data = read_conll(open(trainset, "r", encoding = "utf8"))
    random.shuffle(train_data)
    split = len(train_data) // 20   # on garde 5% du corpus d'entraînement pour servir de corpus de développement)
    dev_data,train_data = train_data[:split], train_data[split:]
    test_data  = read_conll(open(testset, "r", encoding = "utf8"))
    
    return train_data, dev_data, test_data

def get_Universal_corpus_data_predict():
    """
    Fonction pour lire les corpus d'entraînement, de développement et de
    test du tiger corpus (voir lien sur didel).
    """
    trainset = "./data/german/tiger/test/tagging_test_Tiger2Universal_predict.conll"
    testset  = "./universal_treebanks_v2.0/std/de/de-universal-test.conll"
    train_data = read_conll(open(trainset, "r", encoding = "utf8"))
    random.shuffle(train_data)
    split = len(train_data) // 20   # on garde 5% du corpus d'entraînement pour servir de corpus de développement)
    dev_data,train_data = train_data[:split], train_data[split:]
    test_data  = read_conll(open(testset, "r", encoding = "utf8"))
    
    return train_data, dev_data, test_data

def get_intermediaire_corpus_data():
    trainset = "./data/german/tiger/train/german_tiger_train_merged.conll"
    testset  = "./data/german/tiger/test/german_tiger_test_merged.conll"
    train_data = read_conll(open(trainset, "r", encoding = "utf8"))
    random.shuffle(train_data)
    split = len(train_data) // 20   # on garde 5% du corpus d'entraînement pour servir de corpus de développement)
    dev_data,train_data = train_data[:split], train_data[split:]
    test_data  = read_conll(open(testset, "r", encoding = "utf8"))

    return train_data, dev_data, test_data

if __name__ == "__main__":

    # Récupération des données
    #train_data, dev_data, test_data = get_tiger_corpus_data()
    train_data, dev_data, test_data = get_universal_treebank_data("de")

    train_data = list(filter(lambda s: check_projectivity(s[2]), train_data))   # garder uniquement les arbres projectifs

    # Taille des donées (en nombre de phrases)
    print("Train dataset : {} sentences".format(len(train_data)))
    print("Dev dataset   : {} sentences".format(len(dev_data)))
    print("Test dataset  : {} sentences".format(len(test_data)))

    parser = ArcHybridParser(dp_features) # ArcHybrid Parsing algorithm

    print("Training ...")
    train_dependency_parser(parser, train_data, dev_data, 5)            # entraînement, 5 : nombre d'itérations sur les données (typiquement entre 5 et 20)
    print("UAS on test : ", test_dependency_parser(parser, test_data))  # test parser on test dataset
