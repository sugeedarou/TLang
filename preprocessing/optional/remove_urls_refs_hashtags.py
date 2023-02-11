import re
import pandas as pd

def cleanup_dataset(path):
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        text = row['text']
        text = re.sub(r'http\S+', '', text) # no urls
        # text = re.sub(r'@\S*', '', text) # no user refs
        # text = re.sub(r'#\S+', '', text) # no hashtags
        if text != '':
            df.at[index, 'text'] = text
        else:
            df.drop(index, inplace=True)
    df.to_csv(path, sep='\t', index=False)

cleanup_dataset('data/processed/train_val.tsv')
cleanup_dataset('data/processed/test.tsv')
print('done')
