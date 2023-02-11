import pandas as pd
import os
import tempfile

replace_dict = {
    'sr': 'scb', 'hr': 'scb', 'bs': 'scb', # merge serbian, croatian, bosnian
    'no': 'nds', 'da': 'nds', 'sv': 'nds', # merge norwegian, danish, swedish
    'hi-Latn': 'hl', 'ur': 'hl' # merge (latinized) hindu, urdu
}


def merge_simliar_languages(path):
    df = pd.read_csv(path, sep='\t')
    df.replace({'lang': replace_dict}, inplace=True)
    df.to_csv(path, sep='\t', index=False)

merge_simliar_languages('data/processed/train_val.tsv')
merge_simliar_languages('data/processed/test.tsv')

with open('data/langs.tsv', 'r+', encoding='utf-8', newline='') as f:
    langs = f.read().split('\n')[:-1]
    print(langs)
    
    for i in range(len(langs)):
        lang = langs[i]
        if lang in replace_dict:
            lang = replace_dict[lang]
        langs[i] = lang

    f.seek(0)
    f.write('\n'.join(set(langs))+'\n')
    f.truncate()