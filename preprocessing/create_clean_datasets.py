from csv import DictReader, DictWriter
import re
import emoji

used_langs = ['am', 'ar', 'bg', 'bn', 'bo', 'bs', 'ca', 'ckb', 'cs', 'cy', 'da', 'de', 'dv', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hi-Latn', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'km', 'kn', 'ko', 'lo', 'lt', 'lv', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']

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


# create_clean_dataset('../data/raw/sampled_data.csv', '../data/train_val.csv')
create_clean_dataset('../data/raw/recall_data.csv', '../data/test.csv')
print('done')