import torch
import csv
import torch.nn.functional as Fun

from settings import *


class Dataset(torch.utils.data.Dataset):

    class_names = ['lt', 'cy', 'da', 'ca', 'en', 'hy', 'uk', 'bg', 'nl', 'no', 'ht', 'vi', 'ko', 'fi', 'lv', 'th', 'sr', 'tl', 'eu', 'de', 'si', 'ta', 'bn', 'ur', 'fa', 'is', 'pt', 'ro', 'ar', 'km', 'pl', 'mr', 'hi', 'ne', 'es', 'ja', 'sv', 'et', 'tr', 'ru', 'cs', 'hu', 'it', 'sl', 'fr', 'el']
    class_count = len(class_names)
    characters = [l.strip() for l in open(CHARACTERS_PATH, 'r', encoding='utf-8')]
    characters_count = len(characters)

    def __init__(self, path):
        super().__init__()

        def load_data(path):
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                return [l for l in reader]

        self.ds = load_data(path)
        self.n_records = len(self.ds)

    def __getitem__(self, i):
        def indexOrLast(c):
            try:
                return Dataset.characters.index(c)
            except ValueError:
                return Dataset.characters_count
    
        id, lang, text = self.ds[i]
        lang = Dataset.class_names.index(lang)
        text = [indexOrLast(c) for c in text]
        text = text[:TWEET_MAX_CHARACTERS]
        text = Fun.one_hot(torch.tensor(text), num_classes=Dataset.characters_count + 1).float()
        return id, lang, text

    def __len__(self):
        return self.n_records


# ds = Dataset(TRAIN_VAL_PATH)
# id, lang, text = ds.__getitem__(1)
# print(text)
