from csv import DictReader, DictWriter
import re
import emoji

clean_ids = [line.split(' ')[1].strip() for line in open('data/sample_clean.tsv', 'r')]

def remove_emojis(s):
    return emoji.replace_emoji(s, replace='')

out = open('data/data_clean.csv', 'w', encoding='utf-8', newline='')
with open('data/data.csv', 'r', encoding='utf-8', newline='') as f:
     reader = DictReader(f, delimiter=',')
     fieldnames = ['id', 'lang', 'text']
     writer = DictWriter(out, fieldnames=fieldnames)
     writer.writeheader()
     i = 0
     for row in reader:
        _id = row['id']
        if not _id in clean_ids:
            continue

        text = row['text']
        text = text.replace('\r', '').replace('\n', '') # new lines to space
        text = re.sub(r'http\S+', '', text) # no urls
        text = re.sub(r'@\S+', '', text) # no user refs
        text = re.sub(r'#\S+', '', text) # no hashtags
        text = remove_emojis(text) # no emojis
        text = re.sub('  ', ' ', text) # no double whitespaces
        text = text.strip() # no trailing whitespaces

        if text == '':
            continue

        writer.writerow({
            'id': _id,
            'lang': row['lang'],
            'text': text
        })
        # if i > 1000:
        #     break
        i += 1
out.close()

print('done')