import torch
import csv
import torch.nn.functional as Fun

from settings import *


class Dataset(torch.utils.data.Dataset):

    class_names = ['sl', 'fa', 'nl', 'uk', 'km', 'sr', 'da', 'ja', 'zh-CN', 'zh-TW', 'hu', 'ro', 'hr', 'pt', 'fr', 'ur', 'eu', 'fi', 'bn', 'ca', 'cs', 'lv', 'de', 'en', 'es', 'si', 'ru', 'tl', 'no', 'el', 'lt', 'ht', 'th', 'sv', 'ta', 'ar', 'hi-Latn', 'dv', 'cy', 'he', 'ne', 'vi', 'tr', 'ko', 'pl', 'sk', 'it', 'bs', 'id', 'hi', 'mr', 'hy', 'is', 'et', 'bg']
    class_count = len(class_names)
    characters = [l.strip() for l in open(CHARACTERS_PATH, 'r', encoding='utf-8')]
    characters_count = len(characters)

    def __init__(self, path):
        super().__init__()

        def load_data(path):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    id, lang, text = row
                    lang = int(lang)
                    text = [int(c) for c in text.split(' ')]
                    text = text[:TWEET_MAX_CHARACTERS]
                    text = torch.tensor(text)
                    text = Fun.one_hot(text, num_classes=Dataset.characters_count + 1).float()
                    data.append([id, lang, text])
            return data

        self.ds = load_data(path)
        self.n_records = len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]

    def __len__(self):
        return self.n_records


# ds = Dataset(TRAIN_VAL_PATH)
# id, lang, text = ds.__getitem__(1)
# print(text)
