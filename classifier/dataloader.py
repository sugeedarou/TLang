import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np

from classifier.twitter_dataset import TwitterDataset
from settings import *

def collate_fn(batch, batch_size_padding=True, padding_type="wrap"):
    '''
        pads a batch to uniform size (all items in the batch have same dimension after)
    :param batch_size_padding:
        - if batch_size_padding: to the max size of the batch
        - else: padding to fixed size TWEET_MAX_CHARACTERS (from settings.py)
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
        if padding_type == "wrap": #padding_size defined on batch_size
            padding_size = [item[2].size(0) for item in batch]
            padding_size = np.max([item[2].size(0) for item in batch]) - padding_size

        else:
            texts = pad_sequence([item[2] for item in batch], batch_first=True)
            return ids, text_lengths, labels, texts

    else:
        padding_size = [TWEET_MAX_CHARACTERS - item[2].size(0) for item in batch]

    if padding_type == "wrap":
        if len(batch[0][2].size()) == 2: #= One hot encoded
            texts = [torch.FloatTensor(np.pad(item[2].detach().numpy(), ((0, padding_size[idx]), (0, 0)), mode="wrap"))
                             for idx, item in enumerate(batch)]
        else:
            texts = [torch.FloatTensor(np.pad(item[2].detach().numpy, (0, padding_size[idx]), mode="wrap"))
                                       for idx, item in enumerate(batch)]
    else:
        if len(batch[0][2].size()) == 2: #= One hot encoded
            texts = ([F.pad(torch.tensor(item[2]), (0,0,0,padding_size[idx]), mode="constant", value=0)
                              for idx, item in enumerate(batch)])
        else: #encoded just by numbers in 1d
            texts = ([F.pad(torch.tensor(item[2]), (0, padding_size[idx]), mode="constant", value=0)
                      for idx, item in enumerate(batch)])

    texts = torch.stack(texts)
    #print("\n \n texts.size", texts.size(), "\n \n")
    return ids, text_lengths, labels, texts

class DataLoader():

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.loader_args = {'batch_size': self.batch_size,
                            'collate_fn': collate_fn,
                            'num_workers': 12,
                            'pin_memory': True}

    def setup(self, stage):
        self.train_ds = TwitterDataset(TRAIN_PATH)
        self.val_ds = TwitterDataset(VAL_PATH)
        self.test_ds  = TwitterDataset(TEST_PATH)


    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          shuffle=True,
                          **self.loader_args)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          **self.loader_args)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds,
                          **self.loader_args)