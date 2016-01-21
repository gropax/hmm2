from nose.tools import *
from io import StringIO
from hmm.conll_corpus import ConllCorpus, read_map


tag_map = StringIO("""
A	B
B	B
C	D
""")

def test_read_map():
    map = read_map(tag_map)
    expected = {'A': 'B', 'B': 'B', 'C': 'D'}
    assert_equal(expected, map)


class TestConllCorpus:
    def setup(self):
        self.corpus = ConllCorpus('test')

    def test_iter(self):
        pass

    def test_map(self):
        pass
