from nose.tools import *
from hmm import read_conll, dp_features, ArcEagerParser


def test_read_conll():
    pass


class TestDependencyParser:
    def setup(self):
        self.parser = ArcEagerParser(dp_features)

    def test_train(self):
        pass

    def test_test(self):
        pass

    def test_predict(self):
        pass
