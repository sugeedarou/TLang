import torch
from csv import DictReader


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()

        def load_data(path):
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = DictReader(f, delimiter=',')
                return list(reader)

        ds = load_data(path)
        self.n_records = len(ds)

    def __getitem__(self, i):
        return self.ds[i]

    def __len__(self):
        return self.n_records

    def get_class_names(self):
        return ['am', 'ar', 'bg', 'bn', 'bo', 'bs', 'ca', 'ckb', 'cs', 'cy', 'da', 'de', 'dv', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hi-Latn', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'ka', 'km', 'kn', 'ko', 'lo', 'lt', 'lv', 'ml', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']

