#!/usr/bin/env python3

import sys
import random

from logging import info, warning

from longestfirst import Tokenizer


# Special character used to mark beginning of string.
BOS_MARKER = '*'

# Characters expected in input protein sequences
SEQUENCE_CHARS = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
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
            info('merging synonym sets for {}'.format(known_synonyms))
            synset = synonym_map[known_synonyms[0]]    # merge to first
            for s in known_synonyms[1:]:
                synset.update(synonym_map[s])
                synonym_map[s] = synset
            info('merged synonym set: {}'.format(synset))
        synset.update(synonyms)
        for s in synonyms:
            synonym_map[s] = synset
    return synonym_map


def make_tokenizer(synonym_map):
    # This is a bit of a hack: we're reusing code for longest-first wordpiece
    # tokenization to find candidates for synonym replacement, so make a
    # vocabulary of all such candidates plus individual characters as
    # continuation wordpieces (prefixed with "##") and add a special sentry
    # character to both vocab and the start of each sequence so that
    # continuation word pieces can match at the (actual) start of sequences.
    # Then any token longer than a single character is in the synonym_map.
    words = synonym_map.keys()
    info('found {} "words": {}...'.format(len(words), list(words)[:5]))
    chars = set(c for s in words for c in s)
    chars.update(SEQUENCE_CHARS)
    info('found {} characters: {}'.format(len(chars), chars))
    vocab = set(
        [BOS_MARKER] +
        ['##'+w for w in words] +
        ['##'+c for c in chars]
    )
    info('created vocab of {} entries'.format(len(vocab)))
    tokenizer = Tokenizer(vocab)
    def tokenize(sequence):
        tokenized = tokenizer.tokenize(BOS_MARKER + sequence)
        tokenized = [t[2:] for t in tokenized[1:]]
        assert ''.join(tokenized) == sequence, 'internal error'
        return tokenized
    return tokenize


def generate_synonyms(fn, synonym_map, tokenize, options):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) != 2:
                raise ValueError('expected 2 tab-separated fields, got {}'
                                 ' on line {} in {}: {}'.format(
                                     len(fields), ln, fn, l))
            labels, sequence = fields
            tokenized = tokenize(sequence)
            if not any (len(c) > 1 for c in tokenized):
                info('No candidates found for {}'.format(sequence))
                continue
            generated = []
            for t in tokenized:
                if len(t) == 1:
                    generated.append(t)
                else:
                    candidates = [c for c in synonym_map[t] if c != t]
                    assert candidates, 'internal error'
                    generated.append(random.choice(candidates))
            generated = ''.join(generated)
            assert generated != sequence, 'internal error'
            print('{}\t{}'.format(labels, generated))


def main(argv):
    args = argparser().parse_args(argv[1:])
    synonym_sets = read_synonym_sets(args.synonyms, args)
    synonym_map = make_synonym_map(synonym_sets)
    tokenize = make_tokenizer(synonym_map)
    for fn in args.file:
        generate_synonyms(fn, synonym_map, tokenize, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
