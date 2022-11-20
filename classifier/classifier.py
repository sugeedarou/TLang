import torch
import torch.nn as nn
import torchmetrics.functional as FM
import pytorch_lightning as pl
import torch.nn.functional as Fun
from torchmetrics import ConfusionMatrix
from sklearn.utils import class_weight
from csv import DictReader

from settings import *
from models.gru import GRUModel
from dataset import Dataset
from cyclic_plateau_scheduler import CyclicPlateauScheduler


class Classifier(pl.LightningModule):

    def __init__(self, batch_size, lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = Dataset.class_count
        self.model = GRUModel(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.lr_scheduler = CyclicPlateauScheduler(initial_lr=self.lr,
                                                   min_improve_factor=0.97,
                                                   lr_patience=0,
                                                   lr_reduce_factor=0.5,
                                                   lr_reduce_metric='val_loss',
                                                   steps_per_epoch=len(Dataset(TRAIN_VAL_PATH)) / batch_size,
                                                   optimizer=self.optimizer)
        # class_weights = self.calculate_class_weights()
        # class_weights = torch.tensor([9.1871e+02, 4.8353e+01, 4.5035e+00, 5.7419e+01, 3.0624e+02, 3.9944e+01,
        #                         1.1484e+02, 8.0822e-02, 4.8353e+01, 8.3519e+01, 6.1247e+01, 7.6559e+01,
        #                         1.8374e+02, 3.4064e-01, 9.4712e-01, 8.3519e+01, 4.5935e+02, 4.3748e+01,
        #                         1.3124e+02, 4.3748e+01, 2.2968e+02, 3.8280e+01, 4.8609e+00, 4.7589e-02,
        #                         1.4436e-01, 9.1871e+02, 8.8850e-01, 3.1355e+00, 7.6559e+01, 1.9972e+01,
        #                         9.1871e+02, 4.5935e+02, 1.8635e+00, 1.4355e+01, 7.6559e+01, 3.8553e-01,
        #                         5.4042e+01, 9.1871e+02, 9.1871e+02, 2.4830e+01, 1.5312e+02, 4.8353e+01,
        #                         1.3671e+00, 2.0236e+00, 9.2799e+00, 9.1871e+02, 2.4897e+00, 2.2968e+02,
        #                         2.9073e-01, 5.4042e+01, 2.2968e+02, 4.5935e+02, 9.1871e+02, 4.5935e+02,
        #                         2.2968e+02])
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
        self.confmat_metric = ConfusionMatrix(num_classes=self.num_classes)
        # self.f1_metric = F1(num_classes=self.num_classes)

    def calculate_class_weights(self):
        train_val_ds = Dataset(TRAIN_VAL_PATH)
        labels = [item[1] for item in train_val_ds]
        class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=range(Dataset.class_count),y=labels)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        print(class_weights)
        return class_weights

    def training_step(self, batch, step_index):
        self.model.train()
        loss = self.calculate_metrics(batch, mode='train')
        self.lr_scheduler.training_step(step_index)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('lr', self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _step_index):
        self.model.eval()
        loss, accuracy = self.calculate_metrics(batch, mode='val')
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.tensor([o['loss'] for o in outputs]))
        self.logger.experiment.add_scalars('losses', {'train_loss': loss}, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        avg_metrics = {key: torch.mean(torch.tensor([o[key] for o in outputs]))
                       for key in outputs[0].keys()}
        self.lr_scheduler.validation_epoch_end(avg_metrics)
        self.logger.experiment.add_scalars('losses', {'val_loss': avg_metrics['val_loss']}, global_step=self.current_epoch)

    def test_step(self, batch, _):
        self.model.eval()
        loss, accuracy = self.calculate_metrics(batch, mode='test')
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)

    def calculate_metrics(self, batch, mode):
        _, _, labels, texts = batch
        out = self.model(texts.permute(1, 0, 2))
        labels_one_hot = Fun.one_hot(labels, num_classes=Dataset.class_count).float()
        loss = self.criterion(out, labels_one_hot)
        preds = out.argmax(1)

        if mode == 'train':
            return loss
        if mode == 'test':
            self.confmat_metric(preds, labels)
        
        accuracy = FM.accuracy(preds, labels)
        
        return loss, accuracy

    def configure_optimizers(self):
        return self.optimizer
