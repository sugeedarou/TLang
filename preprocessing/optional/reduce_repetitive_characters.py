import pandas as pd
import re

def reduce_dataset(path):
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        text = row['text']
        text = re.sub(r'(.)\1{3,}', r'\1\1', text) # replace >3 repeats with 3 repeats
        text = re.sub(' +', ' ', text) # no double whitespaces
        text = text.strip() # no trailing whitespaces
        if text != '':
            df.at[index, 'text'] = text
        else:
            df.drop(index, inplace=True)
    df.to_csv(path, sep='\t', index=False)


def reduce_repetitive_characters():
    reduce_dataset('data/processed/train_val.tsv')
    reduce_dataset('data/processed/test.tsv')
