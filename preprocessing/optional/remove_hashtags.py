import re
import pandas as pd

def update_dataset(path):
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        text = row['text']
        text = re.sub(r'#\S+', '', text) # no hashtags
        if text != '':
            df.at[index, 'text'] = text
        else:
            df.drop(index, inplace=True)
    df.to_csv(path, sep='\t', index=False)

def remove_hashtags():
    update_dataset('data/processed/train_val.tsv')
    update_dataset('data/processed/test.tsv')