from csv import DictReader, DictWriter

characters = [l.strip() for l in open('../data/characters.csv', 'r', encoding='utf-8')]
characters_count = len(characters)
used_langs = ['lt', 'cy', 'da', 'ca', 'en', 'hy', 'uk', 'bg', 'nl', 'no', 'ht', 'vi', 'ko', 'fi', 'lv', 'th', 'sr', 'tl', 'eu', 'de', 'si', 'ta', 'bn', 'ur', 'fa', 'is', 'pt', 'ro', 'ar', 'km', 'pl', 'mr', 'hi', 'ne', 'es', 'ja', 'sv', 'et', 'tr', 'ru', 'cs', 'hu', 'it', 'sl', 'fr', 'el']

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
