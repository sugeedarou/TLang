from csv import DictReader, DictWriter
import random
from math import ceil
import pandas as pd
import numpy as np

def split_train_val(val_percentage, path_in, path_train, path_val):
    random.seed(42)
    f_in = open(path_in, 'r', encoding='utf-8', newline='')
    reader = DictReader(f_in, delimiter=',')
    rows = [row for row in reader]
    random.shuffle(rows)
    rowCount = len(rows)
    fieldnames = ['id', 'lang', 'text']

    df = pd.read_csv(path_in).sort_values(by='lang')
    df_val = df.groupby(["lang"]).sample(frac=0.1, random_state=1, replace=False)
    df_all = df.merge(df_val.drop_duplicates(), on=fieldnames,
                       how='left', indicator=True)
    df_train = df[df_all['_merge'] == 'left_only'] #only the ones which are in df not temp2

    df_train.to_csv(path_train, encoding='utf-8')
    df_val.to_csv(path_val, encoding='utf-8')
    f_in.close()


VAL_PERCENTAGE = 0.1
split_train_val(VAL_PERCENTAGE, '../data/raw/train_val_reduced.csv', '../data/raw/train_reduced.csv',
                '../data/raw/val_reduced.csv')
print('done')