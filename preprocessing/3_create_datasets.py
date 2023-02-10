from csv import DictReader, DictWriter
import html
import pandas as pd

def get_id_to_lang_dict(path):
    id_to_lang = {}
    with open(path, 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            line_split = line.split('\t')
            id = line_split[1].strip()
            id_to_lang[id] = line_split[0]
    return id_to_lang

def create_dataset(id_lang_path, in_path, out_path):
    id_to_lang = get_id_to_lang_dict(id_lang_path)
    with open(in_path, 'r', encoding='utf-8', newline='') as f_in:
        with open(out_path, 'w', encoding='utf-8', newline='') as f_out:
            reader = DictReader(f_in, delimiter=',')
            writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'], delimiter='\t')
            writer.writeheader()
            for row in reader:
                id = row['id']
                text = row['text'].replace('\r', '').replace('\n', '')
                for _ in range(3): # unescape multiple times to fix all entities
                    text = html.unescape(text)
                lang = id_to_lang[id]

                if text == '':
                    continue

                writer.writerow({
                    'id': id,
                    'lang': lang,
                    'text': text
                })
    df = pd.read_csv(out_path, delimiter='\t')
    df = df.sort_values(['lang', 'text'], ascending=[True, True])
    df.to_csv(out_path, sep='\t', index=False)

create_dataset('data/input/uniform_precision.tsv', 'data/input/uniform_precision_data.csv', 'data/input/train_val.tsv')
create_dataset('data/input/recall_oriented.tsv', 'data/input/recall_data.csv', 'data/input/test.tsv')
print('done')