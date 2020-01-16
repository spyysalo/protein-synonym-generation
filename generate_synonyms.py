#!/usr/bin/env python3

import sys
import re
import random
import logging

from datetime import datetime
from logging import info, warning, debug

from longestfirst import AhocorasickTokenizer, TrieTokenizer


# Characters expected in input protein sequences
SEQUENCE_CHARS = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--verbose', default=False, action='store_true')
    ap.add_argument('--trie', default=False, action='store_true')
    ap.add_argument('--seed', default=None, type=int, help='random seed')
    ap.add_argument('synonyms', help='get_synonyms.py output')
    ap.add_argument('file', nargs='+')
    return ap


def read_synonym_sets(fn, options):
    synonym_sets = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) != 2:
                raise ValueError('expected 2 tab-separated fields, got {}'
                                 ' on line {} in {}: {}'.format(
                                     len(fields), ln, fn, l))
            id_, synonyms = fields
            synonyms = synonyms.split()
            synonym_sets.append(synonyms)
    return synonym_sets


def make_synonym_map(synonym_sets):
    synonym_map = {}
    for synonyms in synonym_sets:
        known_synonyms = [s for s in synonyms if s in synonym_map]
        if not known_synonyms:
            synset = set()    # new set
        elif len(known_synonyms) == 1:
            synset = synonym_map[known_synonyms[0]]    # extend existing
        else:
            # TODO: this can "merge" sets with themselves. This is a
            # no-op, but wasted effort.
            debug('merging synonym sets for {}'.format(known_synonyms))
            synset = synonym_map[known_synonyms[0]]    # merge to first
            for s in known_synonyms[1:]:
                synset.update(synonym_map[s])
                synonym_map[s] = synset
            debug('merged synonym set: {}'.format(synset))
        synset.update(synonyms)
        for s in synonyms:
            synonym_map[s] = synset
    return synonym_map


def make_tokenizer(synonym_map, options):
    words = synonym_map.keys()
    info('found {} "words": {}...'.format(len(words), list(words)[:5]))
    chars = set(c for s in words for c in s)
    chars.update(SEQUENCE_CHARS)
    info('found {} characters: {}'.format(len(chars), chars))
    vocab = set(list(words) + list(chars))
    max_len = max(len(w) for w in words)
    min_len = min(len(w) for w in words)
    info('created vocab of {} entries, word lengths {}-{}'.format(
        len(vocab), min_len, max_len))
    if not options.trie:
        tokenizer = AhocorasickTokenizer(vocab)
    else:
        tokenizer = TrieTokenizer(list(chars), vocab)
    def tokenize(sequence):
        tokenized = tokenizer.tokenize(sequence)
        assert ''.join(tokenized) == sequence, 'internal error'
        return tokenized
    return tokenize


def generate_synonyms(fn, synonym_map, tokenize, options):
    split_re = re.compile('(\W+)')
    with open(fn) as f:
        start_time = datetime.now()
        info('start generating for {} at {}'.format(
            fn, start_time.strftime("%H:%M:%S")))
        for ln, l in enumerate(f, start=1):
            l = l.rstrip()
            fields = split_re.split(l)
            start, sequence = fields[:-1], fields[-1]
            tokenized = tokenize(sequence)
            if not any (len(c) > 1 for c in tokenized):
                info('No candidates found for {}'.format(sequence))
                continue
            generated = []
            for t in tokenized:
                if len(t) == 1:
                    generated.append(t)
                else:
                    candidates = sorted([c for c in synonym_map[t] if c != t])
                    assert candidates, 'internal error'
                    generated.append(random.choice(candidates))
            generated = ''.join(generated)
            assert generated != sequence, 'internal error'
            print('{}{}'.format(''.join(start), generated))
        end_time = datetime.now()
        info('finish generating for {} at {} (delta {})'.format(
            fn, end_time.strftime("%H:%M:%S"), end_time-start_time))


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    random.seed(args.seed)
    synonym_sets = read_synonym_sets(args.synonyms, args)
    synonym_map = make_synonym_map(synonym_sets)
    tokenize = make_tokenizer(synonym_map, args)
    for fn in args.file:
        generate_synonyms(fn, synonym_map, tokenize, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
