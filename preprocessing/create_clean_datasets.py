from csv import DictReader, DictWriter
import re
import emoji

def cleanup_text(text):
    text = text.replace('\r', '').replace('\n', '') # new lines to space
    text = re.sub(r'http\S+', '', text) # no urls
    text = re.sub(r'@\S+', '', text) # no user refs
    text = re.sub(r'#\S+', '', text) # no hashtags
    text = emoji.replace_emoji(text, replace='') # no emojis
    text = re.sub('  ', ' ', text) # no double whitespaces
    text = text.strip() # no trailing whitespaces
    return text


def get_id_to_lang_dict(path):
    id_to_lang = {}
    with open(path, 'r', encoding='utf-8', newline='') as f_in:
        for line in f_in:
            line_split = line.split('\t')
            id = line_split[1].strip()
            id_to_lang[id] = line_split[0]
    return id_to_lang

def get_shared_langs(id_lang_train, data_train, id_lang_test, data_test):
    def get_langs_in_dataset(id_lang_path, data_path):
        used_langs = set()
        id_to_lang = get_id_to_lang_dict(id_lang_path)
        reader = DictReader(open(data_path, 'r', encoding='utf-8', newline=''), delimiter=',')
        for row in reader:
            id = row['id']
            lang = id_to_lang[id]
            used_langs.add(lang)
        return used_langs
    
    langs_train = get_langs_in_dataset(id_lang_train, data_train)
    langs_test = get_langs_in_dataset(id_lang_test, data_test)
    return langs_train & langs_test

def create_clean_dataset(id_lang_path, in_path, out_path, allowed_langs):
    id_to_lang = get_id_to_lang_dict(id_lang_path)
    f_in = open(in_path, 'r', encoding='utf-8', newline='')
    f_out = open(out_path, 'w', encoding='utf-8', newline='')
    reader = DictReader(f_in, delimiter=',')
    writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'])
    writer.writeheader()
    i = 0
    for row in reader:
        id = row['id']
        lang = id_to_lang[id]
        if not lang in allowed_langs:
            continue
        text = cleanup_text(row['text'])

        if text == '':
            continue

        writer.writerow({
            'id': id,
            'lang': lang,
            'text': text
        })

        # if i > 300:
        #     break
        # i += 1
    
    f_in.close()
    f_out.close()


shared_langs = get_shared_langs('../data/raw/uniformly_sampled.tsv', '../data/raw/sampled_data.csv', '../data/raw/recall_oriented.tsv', '../data/raw/recall_data.csv')
create_clean_dataset('../data/raw/uniformly_sampled.tsv', '../data/raw/sampled_data.csv', '../data/raw/train_val_no_enc.csv', shared_langs)
create_clean_dataset('../data/raw/recall_oriented.tsv', '../data/raw/recall_data.csv', '../data/raw/test_no_enc.csv', shared_langs)
print(shared_langs)
print('done')