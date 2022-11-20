from csv import DictReader, DictWriter

characters = [l.strip() for l in open('../data/characters.csv', 'r', encoding='utf-8')]
characters_count = len(characters)
used_langs = ['sl', 'fa', 'nl', 'uk', 'km', 'sr', 'da', 'ja', 'zh-CN', 'zh-TW', 'hu', 'ro', 'hr', 'pt', 'fr', 'ur', 'eu', 'fi', 'bn', 'ca', 'cs', 'lv', 'de', 'en', 'es', 'si', 'ru', 'tl', 'no', 'el', 'lt', 'ht', 'th', 'sv', 'ta', 'ar', 'hi-Latn', 'dv', 'cy', 'he', 'ne', 'vi', 'tr', 'ko', 'pl', 'sk', 'it', 'bs', 'id', 'hi', 'mr', 'hy', 'is', 'et', 'bg']

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
