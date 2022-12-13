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
                                                   steps_per_epoch=len(Dataset(TRAIN_PATH)) / batch_size,
                                                   optimizer=self.optimizer)
        # class_weights = self.calculate_class_weights()
        class_weights = torch.tensor([2.2885, 0.4876, 2.0230, 7.4240, 3.0012, 9.0210, 2.3653, 1.9969, 0.0733,
                                    0.2112, 3.7031, 9.4037, 1.0716, 2.1226, 2.0151, 2.3908, 1.3587, 9.4037,
                                    2.2293, 0.4284, 1.8672, 0.1324, 1.8341, 2.0633, 1.9468, 2.3298, 2.4826,
                                    5.3504, 2.0883, 2.9555, 1.9716, 2.0256, 2.6433, 2.5647, 2.0939, 3.5184,
                                    2.0443, 0.4850, 8.3871, 0.8428, 1.9109, 2.7907, 1.8762, 2.2166, 2.0661,
                                    1.4963, 3.5184, 1.6385, 7.2505, 3.6168, 3.3658, 2.8840])
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.confmat_metric = ConfusionMatrix(num_classes=self.num_classes)
        # self.f1_metric = F1(num_classes=self.num_classes)

    def calculate_class_weights(self):
        train_val_ds = Dataset(TRAIN_PATH)
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
