import pandas as pd

chars = {}
samples_per_lang = {}
df = pd.read_csv('data/raw/train_val_reduced.csv').sort_values(by='lang')
for index, row in df.iterrows():
    text = row['text']
    lang = row['lang']
    if not lang in samples_per_lang:
        samples_per_lang[lang] = 0
    samples_per_lang[lang] += 1

    for c in set(text):
        if not lang in chars:
            chars[lang] = {}
        if not c in chars[lang]:
            chars[lang][c] = 0
        chars[lang][c] += 1

for lang in chars:
    for c in chars[lang]:
        chars[lang][c] = round(chars[lang][c] / samples_per_lang[lang] * 10000) / 100
    sorted_lang = dict(sorted(chars[lang].items(), key=lambda x:x[1], reverse=True))
    chars[lang] = sorted_lang

with open('data/characters_lang_usage.tsv', 'w', encoding='utf-8', newline='') as f_out:
    for lang in chars:
        f_out.write(f'\n---{lang}---\n')
        for c in chars[lang]:
            count = chars[lang][c]
            f_out.write(f'{c}\t{count}%\n')

print('done')