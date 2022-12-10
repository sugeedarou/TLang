import pandas as pd

chars = {}
df = pd.read_csv('data/raw/train_val_reduced.csv').sort_values(by='lang')
for index, row in df.iterrows():
    text = row['text']
    for c in set(text):
        if not c in chars:
            chars[c] = 0
        chars[c] += 1

sorted_chars = dict(sorted(chars.items(), key=lambda x:x[1], reverse=True))

with open('data/characters_freq.tsv', 'w', encoding='utf-8', newline='') as f_out:
    for c in sorted_chars:
        count = sorted_chars[c]
        f_out.write(f'{c}\t{count}\n')