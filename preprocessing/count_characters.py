from csv import DictReader

chars = set()

with open('../data/data_clean.csv', 'r', encoding='utf-8', newline='') as f:
     reader = DictReader(f, delimiter=',')
     
     i = 0
     for row in reader:
        text = row['text']
        chars.update(text)

        i += 1
        # if i > 10000:
        #     break

print(len(chars))

# with open('test.txt', 'w', encoding="utf-8") as f:
#     for c in chars:
#         f.write(c + '\n')

# print('done')
