# protein-synonym-generation

Generate "synonymous" sequences based on Interproscan data

## Quickstart

Create synonym data

```
python3 get_synonyms.py --min-count 2 example-data/prot2ipr.tsv > synonyms.tsv
```

Create "synoymous" sequences for examples

```
python3 generate_synonyms.py synonyms.tsv example-data/labeledseq.tsv \
    > generated.tsv
```

## Data

The file `example-data/prot2ipr.tsv` is a three-column tab-separated values
file where the columns are `PROTEIN_ID`, `INTERPRO_ID`, `SEQUENCE`.
