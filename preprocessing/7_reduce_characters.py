import pandas as pd
import hanzidentifier

chars_by_lang = {}
df = pd.read_csv('data/raw/train_val_clean.csv')
for index, row in df.iterrows():
    lang = row['lang']
    text = row['text']
    if not lang in chars_by_lang:
        chars_by_lang[lang] = set()
    chars_by_lang[lang].update(text)

exclusive_chars_by_lang = {}
for lang in chars_by_lang:
    exclusive = chars_by_lang[lang]
    for lang2 in chars_by_lang:
        if lang2 == lang:
            continue
        exclusive = exclusive.difference(chars_by_lang[lang2])
    exclusive_chars_by_lang[lang] = list(exclusive)

def reduce_char(c):
    if hanzidentifier.is_simplified(c):
        return '这'
    if hanzidentifier.has_chinese(c):
        return '這'
    if u'\u3040' <= c <= u'\u30FF' or u'\u31F0' <= c <= u'\u31FF' or 'ｧ' <= c <= 'ﾝ': # japanese letters
        return 'あ'
    if u'\u1100' <= c <= u'\u11FF' or u'\u3130' <= c <= u'\u318F' or u'\uAC00' <= c <= u'\uD7AF': # korean letters
        return '가'
    for lang in exclusive_chars_by_lang:
        chars = exclusive_chars_by_lang[lang]
        if c in chars:
            return chars[0]
    return c


def reduce_characters():
    characters = set()
    with open('data/raw/characters_clean.csv', 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            c = line.replace('\n', '').replace('\r', '')
            c = reduce_char(c)
            characters.add(c)
    characters = list(characters)
    characters.sort()
            
    with open('data/characters.csv', 'w', encoding='utf-8', newline='') as f_out:
        for c in characters:
            f_out.write(c + '\n')
    
    return characters

def reduce_dataset(in_path, out_path):
    df = pd.read_csv(in_path).sort_values(by='lang')
    for index, row in df.iterrows():
        text = row['text']
        text = ''.join([reduce_char(c) for c in text])
        df.at[index, 'text'] = text
    df.to_csv(out_path, index=False)


characters = reduce_characters()
reduce_dataset('data/raw/train_val_clean.csv', 'data/raw/train_val_reduced.csv')
reduce_dataset('data/raw/test_clean.csv', 'data/raw/test_reduced.csv')

print('done')
