import torch
import csv
import torch.nn.functional as Fun

from settings import *


class TwitterDataset(torch.utils.data.Dataset):
    class_names = open(LANGUAGES_PATH, 'r', encoding='utf-8', newline='').read().split('\n')[:-1]
    num_classes = len(class_names)
    characters = [l.strip() for l in open(CHARACTERS_PATH, 'r', encoding='utf-8')]
    num_characters = len(characters)

    def __init__(self, path, tweet_max_characters):
        '''
            translates text to number sequences (size cut by tweet_max_characters)
            and holds all samples as (id, lang, text) in self.ds

            :param path: string -- syspath: where to read the tweets from
            :param tweet_max_characters:  cutoff value for tweet length
        '''
        super().__init__()

        def load_data(path):
            data = []
            row_counter = 0
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)
                for row in reader:
                    id, lang, text = row
                    lang = int(lang)
                    text = [int(c) for c in text.split(' ')]
                    text = text[:tweet_max_characters]
                    text = torch.tensor(text)
                    data.append([id, lang, text])
                    row_counter += 1

                for i in range(BATCH_SIZE - row_counter % BATCH_SIZE):
                    data.append(data[np.random.randint(0, row_counter)])
            return data

        self.ds = load_data(path)
        self.n_records = len(self.ds)

    def __getitem__(self, i):
        '''
        :param i: list index of the sample
        :return: (id, lang, text)
            - some number representing the tweet
            - language code (as number)
            - one hot encoded text, shape (length_tweet, n_encoding_chars)
        '''
        id, lang, text = self.ds[i]
        text = Fun.one_hot(text, num_classes=TwitterDataset.num_characters + 1).float()
        return id, lang, text

    def __len__(self):
        return self.n_records
