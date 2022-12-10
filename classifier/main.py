import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_module import DataModule
from classifier import Classifier
from dataset import Dataset

# disable all debugging (activate only when training a model - and not testing code!)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
# enable cuDNN autotuner when model is trained for many epochs
torch.backends.cudnn.benchmark = True

num_epochs = 16
batch_size = 16
lr = 1e-3

dm = DataModule(batch_size)

def show_confusion_matrix(confmat):
    plt.figure(figsize=(15,15))
    class_names = Dataset.class_names
    df_cm = pd.DataFrame(confmat, index=class_names, columns=class_names).astype(int)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    heatmap = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d", cmap=cmap)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    model = Classifier(batch_size, lr)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=num_epochs,
                         precision=16,
                         num_sanity_val_steps=0,
                        #  resume_from_checkpoint='lightning_logs/version_13\checkpoints\epoch=2-step=4266.ckpt',
                         callbacks=[ModelCheckpoint(monitor='val_loss'),
                                    EarlyStopping(monitor='val_loss', patience=3)])
    
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    confmat = model.confmat_metric.compute()
    show_confusion_matrix(confmat)