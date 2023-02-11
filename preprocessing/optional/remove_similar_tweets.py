import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def reduce_dataset(path, ratio):
    drop_count = 0
    df = pd.read_csv(path, sep='\t')
    last_text = ''
    for index, row in df.iterrows():
        text = row['text']
        s = similarity(text, last_text)
        if s > ratio:
            df.drop(index, inplace=True)
            drop_count += 1
        last_text = text
    df.to_csv(path, sep='\t', index=False)
    print(f'removed {drop_count} similiar tweets')


ratio = 0.9
reduce_dataset('data/processed/train_val.tsv', ratio)
print('done')

