import pandas as pd


exclusive_chars_by_lang = {}

def create_exclusive_chars_by_lang():
    global exclusive_chars_by_lang
    chars_by_lang = {}
    df = pd.read_csv('data/processed/train_val.tsv', sep='\t')
    for _, row in df.iterrows():
        lang = row['lang']
        text = row['text']
        if not lang in chars_by_lang:
            chars_by_lang[lang] = set()
        chars_by_lang[lang].update(text)

    for lang in chars_by_lang:
        exclusive = chars_by_lang[lang]
        for lang2 in chars_by_lang:
            if lang2 == lang:
                continue
            exclusive = exclusive.difference(chars_by_lang[lang2])
        exclusive_chars_by_lang[lang] = list(exclusive)

def reduce_char(c):
    for lang in exclusive_chars_by_lang:
        chars = exclusive_chars_by_lang[lang]
        if c in chars:
            return chars[0]
    return c

def update_dataset(path):
    df = pd.read_csv(path, sep='\t')
    for index, row in df.iterrows():
        text = row['text']
        text = ''.join([reduce_char(c) for c in text])
        df.at[index, 'text'] = text
    df.to_csv(path, sep='\t', index=False)

def update_characters_list():
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

def reduce_distinct_characters():
    create_exclusive_chars_by_lang()
    update_dataset('data/processed/train_val.tsv')
    update_dataset('data/processed/test.tsv')
    update_characters_list()
