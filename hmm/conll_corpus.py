#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def read_map(file):
    map = {}
    for line in file:
        stripped = line.strip()
        if stripped:
            src, dst = stripped.split('\t')
            map[src] = dst
    return map


class ConllCorpus:
    def __init__(self, corpus, tag_map={}):
        self.corpus = corpus
        self.tag_map = tag_map

    def __iter__(self):
        sent = []
        for l in corpus:
            line = l.strip()
            if line:
                fields = line.split('\t')
                word, tag = fields[1], fields[3]
                mapped = self.tag_map.get(tag, tag)
                sent.append((word, mapped))
            else:
                yield sent
                sent = []
