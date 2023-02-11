import pandas as pd
import hanzidentifier

def reduce_char(c):
    if hanzidentifier.is_simplified(c):
        return '这'
    if hanzidentifier.has_chinese(c):
        return '這'
    if u'\u3040' <= c <= u'\u30FF' or u'\u31F0' <= c <= u'\u31FF' or 'ｧ' <= c <= 'ﾝ': # japanese letters
        return 'あ'
    if u'\u1100' <= c <= u'\u11FF' or u'\u3130' <= c <= u'\u318F' or u'\uAC00' <= c <= u'\uD7AF': # korean letters
        return '가'
    return c

def reduce_characters():
    characters = set()
    with open('data/characters.tsv', 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            c = line.replace('\n', '').replace('\r', '')
            c = reduce_char(c)
            characters.add(c)
    characters = list(characters)
    characters.sort()
            
    with open('data/characters.tsv', 'w', encoding='utf-8', newline='') as f_out:
        for c in characters:
            f_out.write(c + '\n')
    
    return characters

def reduce_dataset(path):
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        text = row['text']
        text = ''.join([reduce_char(c) for c in text])
        df.at[index, 'text'] = text
    df.to_csv(path, sep='\t', index=False)


characters = reduce_characters()
reduce_dataset('data/processed/train_val.tsv')
reduce_dataset('data/processed/test.tsv')
print('done')
