from training_callbacks.training_callback import TrainingCallback
import operator

class EarlyStopping(TrainingCallback):

    def __init__(self, monitor, patience, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.patience_counter = 0
        self.last_metric = float('inf') if mode == 'min' else float('-inf')
        
    def epoch_end_callback(self, val_loss, val_metrics):
        if self.monitor == 'loss':
            metric = val_loss
        else:
            metric = val_metrics[self.monitor]
        op = operator.lt if self.mode == 'min' else operator.gt

        if op(metric, self.last_metric):
            self.last_metric = metric
            self.patience_counter = 0
        else:
            if self.patience_counter < self.patience:
                self.patience_counter += 1
            else:
                return ['stop_training']
        return []

