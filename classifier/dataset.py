import torch
import csv
import torch.nn.functional as Fun

from settings import *


class Dataset(torch.utils.data.Dataset):

    class_names = ['am', 'ar', 'bg', 'bn', 'bo', 'bs', 'ca', 'ckb', 'cs', 'cy', 'da', 'de', 'dv', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hi-Latn', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'km', 'kn', 'ko', 'lo', 'lt', 'lv', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']
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
