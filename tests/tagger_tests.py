from nose.tools import *
from hmm import Tagger, de_universal_corpus


class TestTagger:
    def setup(self):
        self.tagger = Tagger()
        train, dev, test = de_universal_corpus()
        self.train = train[:100]
        self.test = test[:100]

    def test_train(self):
        self.tagger.train(self.train)

    def test_test(self):
        pass

    def test_predict(self):
        pass
