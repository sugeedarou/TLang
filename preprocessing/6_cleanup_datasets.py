import re
import emoji
import pandas as pd

def clean_char(c):
    # if (u'\u2000'  <= c <= u'\u2BFF'  or  # different symbols
    #     u'\u1D100' <= c <= u'\u1D1FF' or  # musical symbols
    #     u'\u1D400' <= c <= u'\u1D7FF' or  # math alphanumeric symbols
    #     u'\u20A0' <= c <= u'\u20CF' or  # currency symbols
    #     u'\u0021' <= c <= u'\u0040' or u'\u005B' <= c <= u'\u0060' or u'\u007B' <= c <= u'\u007E' or # punctuation, digits
    #     u'\u00A1' <= c <= u'\u00BB' 
    
    #     # c in '!"#$%&\'()*+,~-./0123456789:;=?@[]\\^_´`}{§$°<>| '):
    #     return ''
    return c

def clean_text(text):
    text = re.sub(r'http\S+', '', text) # no urls
    text = re.sub(r'@\S*', '', text) # no user refs
    text = re.sub(r'#\S+', '', text) # no hashtags
    text = emoji.replace_emoji(text, replace='') # no emojis
    text = re.sub(r'(.)\1{3,}', r'\1\1', text) # replace repetitive characters
    # text = ''.join([clean_char(c) for c in text])
    text = re.sub(' +', ' ', text) # no double whitespaces
    text = text.strip() # no trailing whitespaces
    return text

def cleanup_characters():
    characters = set()
    with open('data/raw/characters_all.csv', 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            c = line.replace('\n', '').replace('\r', '')
            c = emoji.replace_emoji(c, replace='')
            # c = clean_char(c)
            if c != '':
                characters.add(c)
    characters = list(characters)
    characters.sort()
    with open('data/raw/characters_clean.csv', 'w', encoding='utf-8', newline='') as f_out:
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
