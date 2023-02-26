import pandas as pd

replace_dict = {
    'sr': 'scb', 'hr': 'scb', 'bs': 'scb', # merge serbian, croatian, bosnian
    'no': 'nds', 'da': 'nds', 'sv': 'nds', # merge norwegian, danish, swedish
    'hi-Latn': 'hl', 'ur': 'hl' # merge (latinized) hindu, urdu
}

def update_dataset(path):
    df = pd.read_csv(path, sep='\t')
    df.replace({'lang': replace_dict}, inplace=True)
    df.to_csv(path, sep='\t', index=False)

def update_languages_list():
    with open('data/langs.tsv', 'r+', encoding='utf-8', newline='') as f:
        langs = f.read().split('\n')[:-1]
        
        for i in range(len(langs)):
            lang = langs[i]
            if lang in replace_dict:
                lang = replace_dict[lang]
            langs[i] = lang

        langs = list(set(langs))
        langs.sort()

        f.seek(0)
        f.write('\n'.join(langs)+'\n')
        f.truncate()

def merge_simliar_languages():
    update_dataset('data/processed/train_val.tsv')
    update_dataset('data/processed/test.tsv')
    update_languages_list()