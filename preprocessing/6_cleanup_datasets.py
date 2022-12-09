import hanzidentifier
import re
import emoji
import pandas as pd

def clean_char(c):
    if hanzidentifier.is_simplified(c):
        return '这'
    elif hanzidentifier.has_chinese(c):
        return '這'
    if u'\u3040' <= c <= u'\u30FF' or u'\u31F0' <= c <= u'\u31FF' or 'ｧ' <= c <= 'ﾝ': # japanese letters
        return 'あ'
    if u'\u1100' <= c <= u'\u11FF' or u'\u3130' <= c <= u'\u318F' or u'\uAC00' <= c <= u'\uD7AF': # korean letters
        return '가'
    if (u'\u02B0' <= c <= u'\u036F' or # spacing moodifier letters & combining diacritical marks
        u'\u2000' <= c <= u'\u2BFF' or # different symbols
        u'\u1D100' <= c <= u'\u1D1FF' or # musical symbols
        u'\u1D400' <= c <= u'\u1D7FF' or # math alphanumeric symbols
        '!' <= c <= '@' or '[' <= c <= '`' or '{' <= c <= '½' or '！' <= c <= '＠' or
        '［' <= c <= '｀' or '｛' <= c <= '･'): # different symbols, numbers
        return ''
    return emoji.replace_emoji(c, replace='')

def clean_text(text):
    text = re.sub(r'http\S+', '', text) # no urls
    text = re.sub(r'@\S*', '', text) # no user refs
    text = re.sub(r'#\S+', '', text) # no hashtags
    text = emoji.replace_emoji(text, replace='') # no emojis
    text = ''.join([clean_char(c) for c in text])
    text = re.sub(' +', ' ', text) # no double whitespaces
    text = text.strip() # no trailing whitespaces
    return text

def cleanup_characters():
    characters = set()
    with open('data/raw/characters_all.csv', 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            c = line.replace('\n', '').replace('\r', '')
            c = clean_char(c)
            if c != '':
                characters.add(c)
    characters = list(characters)
    characters.sort()
    with open('data/raw/characters.csv', 'w', encoding='utf-8', newline='') as f_out:
        for c in characters:
            f_out.write(c + '\n')

def cleanup_dataset(in_path, out_path):
    df = pd.read_csv(in_path).sort_values(by='lang')
    for index, row in df.iterrows():
        text = row['text']
        text = clean_text(text)
        if text != '':
            df.at[index, 'text'] = text
        else:
            df.drop(index, inplace=True)
    df.to_csv(out_path, index=False)

cleanup_characters()
cleanup_dataset('data/raw/train_val_eliminated.csv', 'data/raw/train_val_clean.csv')
cleanup_dataset('data/raw/test_eliminated.csv', 'data/raw/test_clean.csv')
print('done')
