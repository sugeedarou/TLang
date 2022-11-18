import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix, F1
import torchmetrics.functional as FM
import pytorch_lightning as pl

from settings import *
from models.gru import GRUModel
from dataset import Dataset


class Classifier(pl.LightningModule):

    def __init__(self, batch_size, lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = len(Dataset.get_class_names())
        self.model = GRUModel(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.confmat_metric = ConfusionMatrix(num_classes=self.num_classes)
        self.f1_metric = F1(num_classes=self.num_classes)

    def training_step(self, batch, step_index):
        self.model.train()
        loss, accuracy = self.calculate_metrics(batch, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('lr', self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, step_index):
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
        # TODO: pass batch data through model and calculate metrics
        return None, None

    def configure_optimizers(self):
        return self.optimizer

    # hide v_num in progres bar in terminal window
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict
