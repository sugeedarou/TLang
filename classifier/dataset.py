import torch
import csv
import torch.nn.functional as Fun
from PIL import Image
from pathlib import Path
from torchvision import transforms

from settings import *


class Dataset(torch.utils.data.Dataset):

    class_names = ['am', 'ar', 'bn', 'bs', 'ca', 'ckb', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hi-Latn', 'hr', 'hu', 'hy', 'id', 'it', 'ja', 'ka', 'km', 'kn', 'ko', 'lo', 'lv', 'ml', 'mr', 'my', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd', 'si', 'sr', 'sv', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']
    class_count = len(class_names)
    characters = [l.strip() for l in open(CHARACTERS_PATH, 'r', encoding='utf-8')]
    characters_count = len(characters)

    def __init__(self, path):
        super().__init__()

        def load_data(path):
            self.path = Path(path)
            data = []
            with open(f'{path}.csv', 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    id, lang, text = row
                    lang = int(lang)
                    # text = [int(c) for c in text.split(' ')]
                    text = text[:TWEET_MAX_CHARACTERS]
                    # text = torch.tensor(text)
                    data.append([id, lang, text])
            return data

        self.ds = load_data(path)
        self.n_records = len(self.ds)

    def __getitem__(self, i):
        id, lang, _ = self.ds[i]
        # text = Fun.one_hot(text, num_classes=Dataset.characters_count + 1).float()
        i = Image.open(self.path / f'{id}.png')
        img = transforms.ToTensor()(i).squeeze().transpose(0, 1)
        return id, lang, img

    def __len__(self):
        return self.n_records


# ds = Dataset(TRAIN_VAL_PATH)
# id, lang, text = ds.__getitem__(1)
# print(text)
