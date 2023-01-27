from csv import DictReader, DictWriter
import random
from math import ceil

def split_train_val(val_percentage, path_in, path_train, path_val):
    random.seed(42)
    f_in = open(path_in, 'r', encoding='utf-8', newline='')
    out_train = open(path_train, 'w', encoding='utf-8', newline='')
    out_val = open(path_val, 'w', encoding='utf-8', newline='')
    reader = DictReader(f_in, delimiter=',')
    rows = [row for row in reader]
    random.shuffle(rows)
    rowCount = len(rows)
    fieldnames = ['id', 'lang', 'text']
    writer_train = DictWriter(out_train, fieldnames=fieldnames)
    writer_val = DictWriter(out_val, fieldnames=fieldnames)
    writer_train.writeheader()
    writer_val.writeheader()

    idx_val_stop = ceil(rowCount * val_percentage)
    for i in range(0, idx_val_stop):
        writer_val.writerow(rows[i])
    for i in range(idx_val_stop, rowCount):
        writer_train.writerow(rows[i])
 
    f_in.close()
    out_train.close()
    out_val.close()


VAL_PERCENTAGE = 0.1
split_train_val(VAL_PERCENTAGE, 'data/raw/train_val_reduced.csv', 'data/raw/train_reduced.csv', 'data/raw/val_reduced.csv')
print('done')
