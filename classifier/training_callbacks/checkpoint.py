from training_callback import TrainingCallback
from operator import operator

class EarlyStopping(TrainingCallback):

    def __init__(self, monitor, patience, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.patience_counter = 0
        self.last_metric = float('inf') if mode == 'min' else float('-inf')
        
    def epoch_end_callback(self, val_metrics):
        metric = val_metrics[self.monitor]
        op = operator.lt if self.mode == 'min' else operator.gt

        if op(metric, self.last_metric):
            
        return False

