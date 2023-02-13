import pandas as pd
import emoji

import pandas as pd

def update_characters():
    characters = set()
    with open('data/characters.tsv', 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            c = line.replace('\n', '').replace('\r', '')
            c.lower()
            characters.add(c)
    characters = list(characters)
    characters.sort()
            
    with open('data/characters.tsv', 'w', encoding='utf-8', newline='') as f_out:
        for c in characters:
            f_out.write(c + '\n')
    
    return characters

def cleanup_dataset(path):
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        text = row['text']
        text = text.lower()
        df.at[index, 'text'] = text
    df.to_csv(path, sep='\t', index=False)


update_characters()
cleanup_dataset('data/processed/train_val.tsv')
cleanup_dataset('data/processed/test.tsv')
print('done')
