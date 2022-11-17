import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
import pytorch_lightning as pl

from dataset import Dataset
from settings import *


def collate_fn(batch):
    lengths = torch.tensor([item[0].size(0) for item in batch])
    fbanks = pad_sequence([item[0] for item in batch], batch_first=True)
    labels = pad_sequence([item[1] for item in batch], batch_first=True)
    sentences = [item[2] for item in batch]
    frame_data = (fbanks, lengths)
    return [frame_data, labels, sentences]

class DataModule(pl.LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage):

        def split_train_val(ds):
            train_val_count = len(ds)
            val_count = int(train_val_count * VAL_PERCENTAGE)
            train_count = train_val_count - val_count
            return random_split(ds, [train_count, val_count])

        self.train_ds, self.val_ds = split_train_val(Dataset(TRAIN_VAL_PATH))
        self.test_ds  = Dataset(TEST_PATH)

        self.loader_args = {'batch_size': self.batch_size,
                            'collate_fn': collate_fn,
                            'num_workers': 4,
                            'pin_memory': True}

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

    def get_class_names(self):
        return self.train_ds.get_class_names()