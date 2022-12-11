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
from models.transformer import TransformerModel
from dataset import Dataset
from cyclic_plateau_scheduler import CyclicPlateauScheduler


class Classifier(pl.LightningModule):

    def __init__(self, batch_size, lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = Dataset.class_count
        self.model = GRUModel(self.num_classes)
        # self.model = TransformerModel(self.num_classes, self.device)
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
        class_weights = torch.tensor([ 2.1637,  0.4610,  1.9126,  7.0190,  2.8375,  8.5289, 11.1982,  2.2362,
                                       1.8880,  0.0693,  0.1997,  3.5011,  8.8907,  1.0131,  2.0068,  1.9051,
                                       2.2603,  2.9457,  8.8907,  2.1077,  0.4050,  1.7653,  0.1252,  1.7340,
                                       1.9507,  1.8406,  2.2026,  2.3471,  5.0585,  1.9744,  2.7942,  1.8640,
                                       2.4991,  2.4247,  8.3826,  1.9797,  3.3264,  1.9328,  0.4586,  7.9295,
                                       0.7968,  1.8066,  2.6384,  1.7738,  3.1891,  2.0957,  1.9533,  1.4146,
                                       3.3264,  1.5491,  6.8550,  2.2779,  3.4195,  3.1821,  2.7267])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
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
