from csv import DictReader

def create_characters_list():
     chars = set()
     with open('data/processed/train_val.tsv', 'r', encoding='utf-8', newline='') as f:
          reader = DictReader(f, delimiter='\t')
          
          for row in reader:
               text = row['text']
               chars.update(text)

     chars = list(chars)
     chars.sort()

     with open('data/characters.tsv', 'w', encoding='utf-8') as f:
          for c in chars:
               if c == '':
                    continue
               f.write(c + '\n')
