from models.gru import GRUModel
from trainer import Trainer
from twitter_dataset import TwitterDataset
from utils import disable_debugging
from settings import *

disable_debugging()

if __name__ == '__main__':
    model = GRUModel()
    ds = TwitterDataset()
    trainer = Trainer(model=model,
                      train_ds=TwitterDataset(TRAIN_PATH),
                      val_ds=TwitterDataset(VAL_PATH),
                      test_ds=TwitterDataset(TEST_PATH),
                      max_epochs=100,
                      batch_size=16,
                      lr=1e-3)

    trainer.train()
    trainer.test()