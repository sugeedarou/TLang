import torch
import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.utils import class_weight
import numpy as np

from settings import *


class DataLoader():
    '''
        loads the already preprocessed data batch-wise,
        computes class_weights performes circular padding
    '''
    def __init__(self, dataset, batch_size, tweet_max_characters):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.tweet_max_characters = tweet_max_characters
        self.loader_args = {'batch_size': self.batch_size,
                            'collate_fn': self.collate_fn,
                            'num_workers': 12,
                            'pin_memory': True,
                            'persistent_workers': True}

        self.train_ds = self.dataset(TRAIN_PATH, tweet_max_characters)
        self.val_ds = self.dataset(VAL_PATH, tweet_max_characters)
        self.test_ds = self.dataset(TEST_PATH, tweet_max_characters)

    def train_dataloader(self):
        return td.DataLoader(dataset=self.train_ds,
                          shuffle=True,
                          **self.loader_args)

    def val_dataloader(self):
        return td.DataLoader(dataset=self.val_ds,
                          **self.loader_args)

    def test_dataloader(self):
        return td.DataLoader(dataset=self.test_ds,
                          **self.loader_args)

    def calculate_class_weights(self):
        labels = [item[1] for item in self.train_ds]
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', classes=range(self.num_classes), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        return class_weights

    def collate_fn(self, batch, batch_size_padding=True, padding_type="wrap"):
        '''
            pads a batch to uniform size (all items in the batch have same dimension after)
        :param batch_size_padding:
            - if batch_size_padding: to the max size of the batch
            - else: padding to fixed size self.tweet_max_characters
        :param batch:
            - batch of items of form [tweet_id, label, tweet_content(text)]
                                    #[int_32(?), array_of_ints(?), string]

        :param padding_type: string: type of padding, "wrap" for cyclic padding, else 0 padding
            default "wrap"
        :return: [tweet_id, length_of_each_tweet, label, padded_texts]
        '''

        ids = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        text_lengths = torch.tensor([item[2].size(0) for item in batch])

        if batch_size_padding:
            if padding_type == "wrap":  # padding_size defined on batch_size
                padding_size = [item[2].size(0) for item in batch]
                padding_size = np.max([item[2].size(0)
                                    for item in batch]) - padding_size

            else:
                texts = pad_sequence([item[2] for item in batch], batch_first=True)
                return ids, text_lengths, labels, texts

        else:
            padding_size = [self.tweet_max_characters -
                            item[2].size(0) for item in batch]

        if padding_type == "wrap":
            if len(batch[0][2].size()) == 2:  # = One hot encoded
                texts = [torch.FloatTensor(np.pad(item[2].detach().numpy(), ((0, padding_size[idx]), (0, 0)), mode="wrap"))
                        for idx, item in enumerate(batch)]
            else:
                texts = [torch.FloatTensor(np.pad(item[2].detach().numpy, (0, padding_size[idx]), mode="wrap"))
                        for idx, item in enumerate(batch)]
        else:
            if len(batch[0][2].size()) == 2:  # = One hot encoded
                texts = ([F.pad(torch.tensor(item[2]), (0, 0, 0, padding_size[idx]), mode="constant", value=0)
                        for idx, item in enumerate(batch)])
            else:  # encoded just by numbers in 1d
                texts = ([F.pad(torch.tensor(item[2]), (0, padding_size[idx]), mode="constant", value=0)
                        for idx, item in enumerate(batch)])

        texts = torch.stack(texts)
        return ids, text_lengths, labels, texts
