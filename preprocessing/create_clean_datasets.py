from csv import DictReader, DictWriter
import re
import emoji

used_langs = ['lt', 'cy', 'da', 'ca', 'en', 'hy', 'uk', 'bg', 'nl', 'no', 'ht', 'vi', 'ko', 'fi', 'lv', 'th', 'sr', 'tl', 'eu', 'de', 'si', 'ta', 'bn', 'ur', 'fa', 'is', 'pt', 'ro', 'ar', 'km', 'pl', 'mr', 'hi', 'ne', 'es', 'ja', 'sv', 'et', 'tr', 'ru', 'cs', 'hu', 'it', 'sl', 'fr', 'el']

def cleanup_text(text):
    text = text.replace('\r', '').replace('\n', '') # new lines to space
    text = re.sub(r'http\S+', '', text) # no urls
    text = re.sub(r'@\S+', '', text) # no user refs
    text = re.sub(r'#\S+', '', text) # no hashtags
    text = emoji.replace_emoji(text, replace='') # no emojis
    text = re.sub('  ', ' ', text) # no double whitespaces
    text = text.strip() # no trailing whitespaces
    return text

def create_clean_dataset(in_path, out_path):
    f_in = open(in_path, 'r', encoding='utf-8', newline='')
    f_out = open(out_path, 'w', encoding='utf-8', newline='')
    reader = DictReader(f_in, delimiter=',')
    writer = DictWriter(f_out, fieldnames=['id', 'lang', 'text'])
    writer.writeheader()
    i = 0
    for row in reader:
        if not row['lang'] in used_langs:
            continue

        text = cleanup_text(row['text'])

        if text == '':
            continue

        writer.writerow({
            'id': row['id'],
            'lang': row['lang'],
            'text': text
        })

        # if i > 300:
        #     break
        # i += 1
    
    f_in.close()
    f_out.close()


create_clean_dataset('../data/raw/sampled_data.csv', '../data/train_val.csv')
create_clean_dataset('../data/raw/recall_data.csv', '../data/test.csv')
print('done')