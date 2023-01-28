import torch
import torch.nn.functional as Fun
import torchmetrics.functional as FM
from torchmetrics import ConfusionMatrix
from tqdm import tqdm

from settings import *
from visualizations import show_confusion_matrix


class Trainer():

    def __init__(self, device, model, dataloader, criterion, optimizer, max_epochs=100, batch_size=16, lr=1e-3, lr_scheduler=None, do_checkpoints=True, early_stopping_patience=3, disable_debugging=True):
        # init variables
        self.task = 'multiclass'
        self.device = device
        self.dataloader = dataloader
        self.num_classes = self.dataloader.dataset.num_classes
        self.confmat_metric = ConfusionMatrix(task=self.task, num_classes=self.num_classes)
        if disable_debugging:
            self.disable_debugging()
        self.model = model
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        # self.f1_metric = F1(num_classes=self.dataset.num_classes)
        # optional parameters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.do_checkpoints = do_checkpoints
        self.early_stopping_patience = early_stopping_patience

    def train(self):
        train_loader = self.dataloader.train_dataloader()
        val_loader = self.dataloader.val_dataloader()

        for epoch in range(1, self.max_epochs+1):
            with tqdm(train_loader) as tepoch: 
                # train
                tepoch.set_description(f'Epoch {epoch} / {self.max_epochs}')
                if epoch > 0:
                    tepoch.set_postfix(train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy)
                train_losses = torch.zeros(len(tepoch))
                for i, batch in enumerate(tepoch):
                    loss = self.training_step(batch, i)
                    self.model.zero_grad()
                    loss.backward()
                    train_losses[i] = loss
                    self.optimizer.step()
                train_loss = torch.mean(train_losses).item()
                # validate
                if epoch == self.max_epochs:
                    continue
                with torch.no_grad(), tqdm(val_loader) as tepoch_val: 
                    tepoch_val.set_description('Validating')
                    val_losses = torch.zeros(len(tepoch_val))
                    val_accuracies = torch.zeros(len(tepoch_val))
                    for i, batch in enumerate(tepoch_val):
                        loss, accuracy = self.validation_step(batch, i)
                        val_losses[i] = loss
                        val_accuracies[i] = accuracy
                val_loss = torch.mean(val_losses).item()
                val_accuracy = torch.mean(val_accuracies).item()
                
    def test(self):
        test_loader = self.dataloader.test_dataloader()
        with torch.no_grad(), tqdm(test_loader) as tepoch: 
            tepoch.set_description('Testing')
            test_losses = torch.zeros(len(tepoch))
            test_accuracies = torch.zeros(len(tepoch))
            for i, batch in enumerate(tepoch):
                loss, accuracy = self.validation_step(batch, i)
                test_losses[i] = loss
                test_accuracies[i] = accuracy
            test_loss = torch.mean(test_losses).item()
            test_accuracy = torch.mean(test_accuracies).item()
            tepoch.set_postfix(train_loss=test_loss, test_accuracy=test_accuracy)
            tepoch.refresh()
            confmat = self.confmat_metric.compute()
            show_confusion_matrix(confmat)

    def log(self, metric, val, on_step=False, on_epoch=True, prog_bar=False, logger=False):
        pass

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
        return loss, accuracy

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.tensor([o['loss'] for o in outputs]))
        # self.logger.experiment.add_scalar('losses', {'train_loss': loss}, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        avg_metrics = {key: torch.mean(torch.tensor([o[key] for o in outputs]))
                       for key in outputs[0].keys()}
        self.lr_scheduler.validation_epoch_end(avg_metrics)
        # self.logger.experiment.add_scalar('losses', {'val_loss': avg_metrics['val_loss']}, global_step=self.current_epoch)

    def test_step(self, batch, _):
        self.model.eval()
        loss, accuracy = self.calculate_metrics(batch, mode='test')
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss, accuracy

    def calculate_metrics(self, batch, mode):
        _, _, labels, texts = batch
        texts = texts.to(self.device)
        labels = labels.to(self.device)
        out = self.model(texts)
        labels_one_hot = Fun.one_hot(labels, num_classes=self.num_classes).float().to(self.device)
        loss = self.criterion(out, labels_one_hot)
        preds = out.argmax(1)

        if mode == 'train':
            return loss
        if mode == 'test':
            self.confmat_metric(preds, labels)
        
        accuracy = FM.accuracy(task=self.task, preds=preds, target=labels, num_classes=self.num_classes)
        
        return loss, accuracy

    def disable_debugging(self):
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    def get_gpu_device_if_available():
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using {device} device")
        return device