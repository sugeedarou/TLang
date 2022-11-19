import torch
import torch.nn as nn
import torchmetrics.functional as FM
import pytorch_lightning as pl
import torch.nn.functional as Fun

from settings import *
from models.gru import GRUModel
from dataset import Dataset


class Classifier(pl.LightningModule):

    def __init__(self, batch_size, lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = Dataset.class_count
        self.model = GRUModel(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # self.confmat_metric = ConfusionMatrix(num_classes=self.num_classes)
        # self.f1_metric = F1(num_classes=self.num_classes)

    def training_step(self, batch, step_index):
        self.model.train()
        loss, accuracy = self.calculate_metrics(batch, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, step_index):
        self.model.eval()
        loss, accuracy = self.calculate_metrics(batch, mode='val')
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    # def training_epoch_end(self, outputs):
    #     loss = torch.mean(torch.tensor([o['loss'] for o in outputs]))
    #     self.logger.experiment.add_scalars('losses', {'train_loss': loss}, global_step=self.current_epoch)

    # def validation_epoch_end(self, outputs):
    #     avg_metrics = {key: torch.mean(torch.tensor([o[key] for o in outputs]))
    #                    for key in outputs[0].keys()}
    #     self.logger.experiment.add_scalars('losses', {'val_loss': avg_metrics['val_loss']}, global_step=self.current_epoch)

    def test_step(self, batch, _):
        self.model.eval()
        loss, accuracy = self.calculate_metrics(batch, mode='test')
        # self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)

    def calculate_metrics(self, batch, mode):
        _, _, labels, texts = batch
        out = self.model(texts.permute(1, 0, 2))
        labels_one_hot = Fun.one_hot(labels, num_classes=Dataset.class_count).float()
        loss = self.criterion(out, labels_one_hot)
        preds = out.argmax(1)
        accuracy = FM.accuracy(preds, labels)
        return loss, accuracy

    def remove_padding(self, tensor, lengths):
        return [tensor[i][:lengths[i]] for i in range(tensor.size(0))]

    def configure_optimizers(self):
        return self.optimizer
