import pandas as pd
import emoji

import pandas as pd

def update_characters():
    characters = set()
    with open('data/characters.tsv', 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            c = line.replace('\n', '').replace('\r', '')
            if not emoji.is_emoji(c):
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
        text = emoji.replace_emoji(text, replace='') # no emojis
        if text != '':
            df.at[index, 'text'] = text
        else:
            df.drop(index, inplace=True)
    df.to_csv(path, sep='\t', index=False)


update_characters()
cleanup_dataset('data/processed/train_val.tsv')
cleanup_dataset('data/processed/test.tsv')
print('done')
