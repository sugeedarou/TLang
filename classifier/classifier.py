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
        # class_weights = torch.tensor([ 2.0877,  0.4448,  1.8455,  9.5640,  6.7726,  2.7379,  8.2295, 10.8051,
        #                                2.1577,  1.8217,  0.0669,  0.1927,  3.3782,  8.5786,  0.9775,  1.9364,
        #                                1.8383,  2.1810,  2.8423,  7.8203,  8.5786,  2.0337,  0.3908,  1.7033,
        #                                0.1208,  1.6731,  1.8823,  1.7760,  2.1253,  2.2648,  4.8809,  1.9051,
        #                                2.6961,  1.7986,  2.4114,  2.3396,  8.0884,  1.9102,  3.2097,  1.8649,
        #                                0.4425,  7.6512,  0.7689,  2.5458,  1.7116,  2.9306,  3.0771,  2.0221,
        #                                1.8848,  1.3650,  3.2097,  1.4947,  6.6144,  2.1979,  3.2995,  3.0704,
        #                                2.6310])
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
        _, _, labels, img = batch
        out = self.model(img)
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
