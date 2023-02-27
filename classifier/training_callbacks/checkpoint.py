from training_callbacks.training_callback import TrainingCallback
import operator

class ModelCheckpoint(TrainingCallback):

    def __init__(self, monitor, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.last_metric = float('inf') if mode == 'min' else float('-inf')
        
    def epoch_end_callback(self, val_loss, val_metrics):
        if self.monitor == 'loss':
            metric = val_loss
        else:
            metric = val_metrics[self.monitor]
        op = operator.lt if self.mode == 'min' else operator.gt

        if op(metric, self.last_metric):
            return ['save_checkpoint']
        return []

