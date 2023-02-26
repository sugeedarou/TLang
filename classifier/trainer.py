import torch
import torch.nn.functional as Fun
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from torchmetrics import ConfusionMatrix
from tqdm import tqdm
from math import floor
from pathlib import Path
from contextlib import nullcontext
import pandas as pd

from settings import *
from visualizations import show_confusion_matrix


class Trainer():

    def __init__(self, device, model, dataloader, criterion, optimizer, max_epochs=100, batch_size=16, lr=1e-3, lr_scheduler=None, disable_debugging=True, resume_from_checkpoint=None, mixed_precision=False, callbacks=[]):
        # init variables
        self.task = 'multiclass'
        self.device = device
        self.dataloader = dataloader
        self.num_classes = self.dataloader.dataset.num_classes
        self.confmat_metric = ConfusionMatrix(task=self.task, num_classes=self.num_classes)
        if disable_debugging:
            self.disable_debugging()
        self.model = model
        self.model.to(device)
        model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model has {model_total_params} parameters')
        self.criterion = criterion
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision
        # performance metrics
        metric_args = {'task':self.task, 'num_classes':self.num_classes, 'average':'macro'}
        self.val_metrics = MetricCollection({
            'acc': Accuracy(**metric_args).to(device),
            'f1': F1Score(**metric_args).to(device),
        })
        self.test_metrics = MetricCollection({
            'acc': Accuracy(**metric_args).to(device),
            'f1': F1Score(**metric_args).to(device),
            'prec': Precision(**metric_args).to(device),
            'rec': Recall(**metric_args).to(device),
        })
        self.confmat_metric = ConfusionMatrix(task=self.task, num_classes=self.num_classes).to(device)
        # optional parameters
        self.max_epochs = max_epochs
        self.epoch = 1
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.callbacks = callbacks
        self.log_version = self.get_log_version()
        self.log_dir = Path(f'runs/version_{self.log_version}')
        self.tb_writer = SummaryWriter(self.log_dir)
        self.tb_writer.add_custom_scalars({
            "metrics": {
                "loss": ["Multiline", ["loss/train", "loss/val"]],
                "accuracy": ["Multiline", ["accuracy/val", "accuracy/test"]],
                "f1": ["Multiline", ["f1/val", "f1/test"]],
                "prec": ["Multiline", ["prec/test"]],
                "rec": ["Multiline", ["rec/test"]],
            },
        })
        if resume_from_checkpoint:
            self.load_model_from_checkpoint(resume_from_checkpoint)
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def train(self):
        train_loader = self.dataloader.train_dataloader()
        val_loader = self.dataloader.val_dataloader()
        train_loss = None

        for epoch in range(self.epoch, self.max_epochs+1):
            self.epoch = epoch
            with tqdm(train_loader) as train_tepoch: 
                # train
                train_tepoch.set_description(f'Epoch {epoch} / {self.max_epochs}')
                if train_loss:
                    log_str_train = self.get_log_metrics_str('train', train_loss, [])
                    log_str_val   = self.get_log_metrics_str('val', val_loss, val_metrics)
                    train_tepoch.set_postfix_str(f'{log_str_train}, {log_str_val}')
                train_loss = self.training_epoch(train_tepoch)
                # validate
                with tqdm(val_loader) as val_tepoch: 
                    val_tepoch.set_description('Validating')
                    val_loss, val_metrics = self.validation_epoch(val_tepoch)
                # callbacks
                stop_training = False
                save_checkpoint = False
                for callback in self.callbacks:
                    result = callback.epoch_end_callback(val_loss, val_metrics)
                    if "stop_training" in result:
                        stop_training = True
                    if "save_checkpoint" in result:
                        save_checkpoint = True
                if save_checkpoint:
                    self.save_checkpoint()
                if stop_training:
                    return

    def update_gradient_with_loss(self, loss):
        if self.mixed_precision:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def training_epoch(self, data):
         # mixed precision for faster training
        losses = torch.zeros(len(data))
        for i, batch in enumerate(data):
            self.model.zero_grad()
            _, loss = self.training_step(batch, i)
            losses[i] = loss
            self.update_gradient_with_loss(loss)
        loss = torch.mean(losses).item()
        self.tb_writer.add_scalar('loss/train', loss, self.epoch)
        return loss

    def training_step(self, batch, step_index):
        _, _, labels, texts = batch
        texts = texts.to(self.device)
        labels = labels.to(self.device)
        loss = self.predict_with_model(texts, labels)
        if self.lr_scheduler:
            self.lr_scheduler.training_step(step_index)
        return loss

    def validation_epoch(self, data):
        with torch.no_grad():
            losses = torch.zeros(len(data))
            for i, batch in enumerate(data):
                losses[i] = self.validation_step(batch, i)
            loss = torch.mean(losses).item()
            if self.lr_scheduler:
                self.lr_scheduler.validation_epoch_end({'val_loss': loss})
            metrics = self.val_metrics.compute()
            self.val_metrics.reset()
            self.tb_writer.add_scalar('loss/val', loss, self.epoch)
            self.tb_writer.add_scalar('accuracy/val', metrics['acc'], self.epoch)
            self.tb_writer.add_scalar('f1/val', metrics['f1'], self.epoch)
            return loss, metrics

    def validation_step(self, batch, _step_index):
        _, _, labels, texts = batch
        texts = texts.to(self.device)
        labels = labels.to(self.device)
        out, loss = self.predict_with_model(texts, labels)
        preds = out.argmax(1)
        self.val_metrics(preds, labels)
        return loss
    
    def test(self):
        checkpoint_path = self.log_dir / 'model.pth'
        if checkpoint_path.exists():
            self.load_model_from_checkpoint(checkpoint_path)
        test_loader = self.dataloader.test_dataloader()
        with torch.no_grad(), tqdm(test_loader) as tepoch: 
            tepoch.set_description('Testing')
            loss, metrics = self.test_epoch(tepoch)
            log_str   = self.get_log_metrics_str('test', loss, metrics)
            print(log_str)
            confmat = self.confmat_metric.compute().cpu()
            show_confusion_matrix(confmat)

    def test_epoch(self, data):
        with torch.no_grad():
            losses = torch.zeros(len(data))
            for i, batch in enumerate(data):
                losses[i] = self.test_step(batch, i)
            loss = torch.mean(losses).item()
            metrics = self.test_metrics.compute()
            self.test_metrics.reset()
            print(metrics)
            self.tb_writer.add_scalar('loss/test', loss, self.epoch)
            self.tb_writer.add_scalar('accuracy/test', metrics['acc'], self.epoch)
            self.tb_writer.add_scalar('f1/test', metrics['f1'], self.epoch)
            return loss, metrics

    def test_step(self, batch, _):
        ids, _, labels, texts = batch
        texts = texts.to(self.device)
        labels = labels.to(self.device)
        out, loss = self.predict_with_model(texts, labels)
        preds = out.argmax(1)

        self.test_metrics(preds, labels)
        self.confmat_metric(preds, labels)

        return loss

    def predict_with_model(self, texts, labels):
        labels_one_hot = Fun.one_hot(labels, num_classes=self.num_classes).float().to(self.device)
        # mixed precision
        with torch.cuda.amp.autocast() if self.mixed_precision else nullcontext():
            out = self.model(texts)
        loss = self.criterion(out, labels_one_hot)
        return out, loss

    def write_test_predictions_to_file(self):
        df = pd.DataFrame(self.test_predictions, columns=['id', 'lang', 'text', 'pred'])
        df.to_csv('predictions.tsv', sep='\t', index=False)

    def disable_debugging(self):
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)

    def save_checkpoint(self):
        torch.save({'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
	        self.log_dir / 'model.pth')

    def load_model_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print('loading model from checkpoint {checkpoint_path}')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] + 1

    def get_log_version(self):
        if not Path('runs').exists():
            return 0
        return len([p for p in Path('runs').iterdir()])

    def get_gpu_device_if_available():
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using {device} device")
        return device

    def get_log_metrics_str(self, type_name, loss, metrics):
        log_metrics_str = f'{type_name}_loss={self.format_num_for_print(loss)}'
        for i, metric in enumerate(metrics):
            log_metrics_str += f', {type_name}_{metric}={self.format_num_for_print(metrics[metric].item())}'
        return log_metrics_str

    def format_num_for_print(self, n):
        if n > 1:
            return round(n*100) / 100
        return round(n*1000) / 1000