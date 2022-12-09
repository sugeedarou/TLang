from csv import DictReader

chars = set()
with open('data/raw/train_val_eliminated.csv', 'r', encoding='utf-8', newline='') as f:
     reader = DictReader(f, delimiter=',')
     
     for row in reader:
        text = row['text']
        chars.update(text)

chars = list(chars)
chars.sort()

with open('data/raw/characters_all.csv', 'w', encoding="utf-8") as f:
     for c in chars:
          if c == '':
               continue
          f.write(c + '\n')

print('done')
