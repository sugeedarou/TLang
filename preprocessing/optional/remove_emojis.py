import pandas as pd
import emoji

import pandas as pd
import re

def cleanup_dataset(path):
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        text = row['text']
        text = emoji.replace_emoji(text, replace='') # no emojis
        df.at[index, 'text'] = text
    df.to_csv(path, sep='\t', index=False)


cleanup_dataset('data/processed/train_val.tsv')
cleanup_dataset('data/processed/test.tsv')
print('done')
