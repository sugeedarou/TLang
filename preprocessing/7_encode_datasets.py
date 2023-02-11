from csv import DictReader, DictWriter

characters = [l.replace('\n', '') for l in open('data/characters.tsv', 'r', encoding='utf-8')]
characters_count = len(characters)
used_langs = open('data/langs.tsv', 'r', encoding='utf-8', newline='').read().split('\n')[:-1]

def encode_dataset(in_path, out_path):
    def indexOrLast(c):
            try:
                return characters.index(c)
            except ValueError:
                return characters_count
    
    f_in = open(in_path, 'r', encoding='utf-8', newline='')
    f_out = open(out_path, 'w', encoding='utf-8', newline='')
    reader = DictReader(f_in, delimiter='\t')
    writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'], delimiter='\t')
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


encode_dataset('data/processed/train.tsv', 'data/train.tsv')
encode_dataset('data/processed/val.tsv', 'data/val.tsv')
encode_dataset('data/processed/test.tsv', 'data/test.tsv')
print('done')