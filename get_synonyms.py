#!/usr/bin/env python3

# Helper for generate_synonyms.py. Given a TSV file with (protein-id,
# interpro-id, sequence), groups sequences sharing the same
# interpro-id and outputs a TSV file with (interpro-id, sequence-list)
# where sequence-list is a space-separated list of sequences.

import sys

from collections import defaultdict, Counter


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--min-length', default=2, type=int,
                    help='Minimum length of sequences to include')
    ap.add_argument('--min-count', default=None, type=int,
                    help='Minimum occurrence number of sequences to include')
    ap.add_argument('file', nargs='+')
    return ap


def get_synonyms(fn, sequence_count_by_id, args):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            fields = l.split('\t')
            if len(fields) != 3:
                raise ValueError('expected 3 tab-separated fields, got {}'
                                 ' on line {} in {}: {}'.format(
                                     len(fields), ln, fn, l))
            protein_id, interpro_id, sequence = fields
            if len(sequence) >= args.min_length:
                sequence_count_by_id[interpro_id][sequence] += 1


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.min_length < 2:
        raise ValueError('--min-length must be 2 or greater')
    sequence_count_by_id = defaultdict(Counter)
    for fn in args.file:
        get_synonyms(fn, sequence_count_by_id, args)
    for id_, counts in sequence_count_by_id.items():
        if args.min_count is not None:
            counts = { s: c for s, c in counts.items() if c >= args.min_count }
        if len(counts) < 2:
            continue    # no synonyms
        print('{}\t{}'.format(id_, ' '.join(counts.keys())))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

