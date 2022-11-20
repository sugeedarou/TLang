from csv import DictReader, DictWriter

characters = [l.strip() for l in open('../data/characters.csv', 'r', encoding='utf-8')]
characters_count = len(characters)
used_langs = ['is', 'uk', 'fa', 'ka', 'ru', 'es', 'pa', 'nl', 'hy', 'tr', 'zh-CN', 'et', 'bs', 'it', 'vi', 'lo', 'mr', 'no', 'eu', 'my', 'ur', 'el', 'tl', 'kn', 'en', 'ko', 'lv', 'sr', 'sv', 'ml', 'fi', 'ps', 'th', 'pt', 'si', 'id', 'zh-TW', 'ht', 'bn', 'ta', 'gu', 'cs', 'lt', 'ca', 'sl', 'hi', 'ne', 'ar', 'hu', 'ro', 'dv', 'sd', 'de', 'ja', 'km', 'pl', 'hr', 'te', 'cy', 'ckb', 'he', 'fr', 'da', 'sk', 'bg', 'am', 'hi-Latn']

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


encode_dataset('../data/raw/train_val_no_enc.csv', '../data/train_val.csv')
encode_dataset('../data/raw/test_no_enc.csv', '../data/test.csv')
print('done')
