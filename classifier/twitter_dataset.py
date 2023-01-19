import torch
import csv
import torch.nn.functional as Fun

from settings import *


class TwitterDataset(torch.utils.data.Dataset):

    class_names = ['am', 'ar', 'bn', 'ca', 'ckb', 'cs', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hl', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'km', 'kn', 'ko', 'lo', 'lv', 'ml', 'mr', 'my', 'nds', 'ne', 'nl', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'scb', 'sd', 'si', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'vi', 'zh-CN', 'zh-TW']
    num_classes = len(class_names)
    characters = [l.strip() for l in open(CHARACTERS_PATH, 'r', encoding='utf-8')]
    num_characters = len(characters)

    def __init__(self, path, tweet_max_characters):
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
                    text = text[:tweet_max_characters]
                    text = torch.tensor(text)
                    data.append([id, lang, text])
            return data

        self.ds = load_data(path)
        self.n_records = len(self.ds)

    def __getitem__(self, i):
        id, lang, text = self.ds[i]
        text = Fun.one_hot(text, num_classes=TwitterDataset.num_characters + 1).float()
        return id, lang, text

    def __len__(self):
        return self.n_records
