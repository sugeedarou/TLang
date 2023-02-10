import pandas as pd
import os
import tempfile


def merge_simliar_languages(in_path, out_path):
    df = pd.read_csv(in_path, delimiter='\t')
    df.replace({'lang': {
        'sr': 'scb', 'hr': 'scb', 'bs': 'scb', # merge serbian, croatian, bosnian
        'no': 'nds', 'da': 'nds', 'sv': 'nds', # merge norwegian, danish, swedish
        'hi-Latn': 'hl', 'ur': 'hl' # merge (latinized) hindu, urdu
    }}, inplace=True)
    df.to_csv(out_path, sep='\t', index=False)

merge_simliar_languages('data/processed/train_val.tsv', 'data/processed/train_val.tsv')
merge_simliar_languages('data/processed/test.tsv', 'data/processed/test.tsv')