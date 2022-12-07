test_in = open('../data/raw/test_no_enc.csv', 'r', encoding='utf-8', newline='')
train_val_out = open('../data/raw/train_val_eliminated.tsv', 'w', encoding='utf-8', newline='')
test_out = open('../data/raw/test_eliminated.tsv', 'w', encoding='utf-8', newline='')

langs = {}
i=0
with open('../data/raw/train_val_no_enc.csv', 'r', encoding='utf-8', newline='') as train_val_in:
    next(train_val_in)
    for line in train_val_in:
        line_split = line.split(',')
        lang = line_split[1]
        if not lang in langs:
            langs[lang] = 1
        else:
            langs[lang] += 1
        
        i += 1
        if i > 100:
            break

print(langs)