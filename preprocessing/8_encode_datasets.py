from csv import DictReader, DictWriter

characters = [l.replace('\n', '') for l in open('data/characters.csv', 'r', encoding='utf-8')]
characters_count = len(characters)
used_langs = ['am', 'ar', 'bn', 'ca', 'ckb', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hi-Latn', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'km', 'kn', 'ko', 'lo', 'lv', 'ml', 'mr', 'my', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'scb', 'sd', 'si', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']

def encode_dataset(in_path, out_path):
    def indexOrLast(c):
            try:
                return characters.index(c)
            except ValueError:
                return characters_count
    
    f_in = open(in_path, 'r', encoding='utf-8', newline='')
    f_out = open(out_path, 'w', encoding='utf-8', newline='')
    reader = DictReader(f_in, delimiter=',')
    writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'])
    writer.writeheader()

    for row in reader:
        text = " ".join([str(indexOrLast(c)) for c in row['text']])
        lang = used_langs.index(row['lang'])

        writer.writerow({
            'id': row['id'],
            'lang': lang,
            'text': text
        })
    
    f_in.close()
    f_out.close()


encode_dataset('data/raw/train_val_reduced.csv', 'data/train_val.csv')
encode_dataset('data/raw/test_reduced.csv', 'data/test.csv')
print('done')
