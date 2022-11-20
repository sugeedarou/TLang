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


class Classifier(pl.LightningModule):

    def __init__(self, batch_size, lr):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = Dataset.class_count
        self.model = GRUModel(self.num_classes)
        class_weights = torch.tensor([4.6198e+01, 3.3204e+01, 2.5916e+01, 1.5399e+01, 5.5243e-02, 5.3127e+02,
                        2.9515e+01, 5.9030e+01, 4.1998e+00, 3.2198e+01, 9.4870e+00, 5.0597e+01,
                        2.3353e+00, 3.0358e+01, 3.1251e+01, 2.2044e+00, 2.1251e+02, 1.7249e+00,
                        3.3204e+01, 4.4458e+00, 1.0625e+03, 9.6595e+01, 1.3282e+02, 8.1734e+01,
                        5.5923e+01, 6.2503e+01, 3.8401e-01, 3.2198e+01, 4.4476e-01, 3.5418e+02,
                        7.0367e+00, 2.6564e+02, 1.4965e+01, 1.7709e+02, 1.6065e-01, 9.3353e-02,
                        1.2213e+01, 8.4329e+00, 1.4148e+00, 1.0700e+00, 3.1251e+01, 3.2198e+01,
                        2.1509e+00, 9.6595e+01, 1.0139e+00, 2.7245e+01])
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.confmat_metric = ConfusionMatrix(num_classes=self.num_classes)
        # self.f1_metric = F1(num_classes=self.num_classes)

    def calculate_class_weights(self):
        train_val_ds = Dataset(TRAIN_VAL_PATH)
        labels = [item[1] for item in train_val_ds]
        class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=range(Dataset.class_count),y=labels)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        print(class_weights)
        return class_weights

    def training_step(self, batch, _step_index):
        self.model.train()
        loss, accuracy = self.calculate_metrics(batch, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, _step_index):
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
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)

    def calculate_metrics(self, batch, mode):
        _, _, labels, texts = batch
        out = self.model(texts.permute(1, 0, 2))
        labels_one_hot = Fun.one_hot(labels, num_classes=Dataset.class_count).float()
        loss = self.criterion(out, labels_one_hot)
        preds = out.argmax(1)
        accuracy = FM.accuracy(preds, labels)
        if mode == 'test':
            self.confmat_metric(preds, labels)
        
        return loss, accuracy

    def configure_optimizers(self):
        return self.optimizer
