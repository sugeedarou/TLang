import torch
import torch.nn as nn

from cyclic_plateau_scheduler import CyclicPlateauScheduler
from models.gru import GRUModel
from trainer import Trainer
from twitter_dataset import TwitterDataset
from dataloader import DataLoader
from settings import *
from training_callbacks.checkpoint import ModelCheckpoint
from training_callbacks.early_stopping import EarlyStopping

batch_size = 32
lr = 1e-4


if __name__ == '__main__':
    device = Trainer.get_gpu_device_if_available()

    # class_weights = torch.tensor([2.2885, 0.4876, 2.0230, 7.4240, 3.0012, 9.0210, 2.3653, 1.9969, 0.0733,
    #                           0.2112, 3.7031, 9.4037, 1.0716, 2.1226, 2.0151, 2.3908, 1.3587, 9.4037,
    #                           2.2293, 0.4284, 1.8672, 0.1324, 1.8341, 2.0633, 1.9468, 2.3298, 2.4826,
    #                           5.3504, 2.0883, 2.9555, 1.9716, 2.0256, 2.6433, 2.5647, 2.0939, 3.5184,
    #                           2.0443, 0.4850, 8.3871, 0.8428, 1.9109, 2.7907, 1.8762, 2.2166, 2.0661,
    #                           1.4963, 3.5184, 1.6385, 7.2505, 3.6168, 3.3658, 2.8840]).to(device)  # recalculate with dataloader.calculate_class_weights
    
    model = GRUModel(TwitterDataset.num_classes)
    dataloader = DataLoader(dataset=TwitterDataset,
                            batch_size=batch_size, tweet_max_characters=128)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(device=device,
                      model=model,
                      dataloader=dataloader,
                      criterion=nn.CrossEntropyLoss(),
                      optimizer=optimizer,
                      batch_size=batch_size,
                      lr=lr,
                      lr_scheduler=None,
                      max_epochs=10,
                    #   resume_from_checkpoint='version_2/model.pth',
                      callbacks=[ModelCheckpoint(monitor='loss'),
                                 EarlyStopping(monitor='loss', patience=3)])

    trainer.train()
    trainer.test()

"""CyclicPlateauScheduler(optimizer=optimizer,
                                                          initial_lr=lr,
                                                          min_improve_factor=0.97,
                                                          lr_patience=0,
                                                          lr_reduce_factor=0.5,
                                                          steps_per_epoch=len(dataloader.train_ds) / batch_size),"""