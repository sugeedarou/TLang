import torch
import torch.nn as nn
from settings import * #this has to be this bad as it gets called from toplevel
from twitter_dataset import TwitterDataset as Dataset


class GRUModel_SVM(nn.Module):
    def __init__(self, output_size, used_kernel="rbf", init_gamma=1.0,
                  trainable_gamma=True):
        super().__init__()
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=Dataset.characters_count+1,
                          hidden_size=256,
                          num_layers=3,
                          bidirectional=True, dropout=0.0,
                          batch_first=True)

        self.dropout = nn.Dropout(0.0)

        self.SVM = MultiCatKernelSVM(self.rnn.hidden_size*2,
                                     self.output_size,
                                     kernel=used_kernel,
                                     gamma_init=init_gamma,
                                     train_gamma=trainable_gamma)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.dropout(out[:, -1])
        out = self.SVM.forward(out)
        return out


class Transphere_GRUModel_CNN(nn.Module):
    def __init__(self, output_size, recur_model=None, kernelsize=4, stride=1):
        super().__init__()
        if recur_model is None:
            print("Transphere_GRUModel_SVM did not get a model to base itself on")
            exit()

        self.output_size = output_size
        self.rnn = recur_model.rnn
        for p in self.rnn.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(0.2)
        self.cnn = nn.Conv1d(BATCH_SIZE, BATCH_SIZE, kernel_size=kernelsize, stride=stride)
        cnn_out_size = (2 * self.rnn.hidden_size - kernelsize) // stride + 1
        self.fc = nn.Linear(cnn_out_size, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.dropout(out[:, -1])
        out = self.cnn.forward(out)
        out = self.fc(out)
        return out


class Transphere_GRUModel_CNN_2layer(Transphere_GRUModel_CNN):
    '''
        transphere gru_cnn with 1 extra cnn layer,
        changes: added 2nd cnn layer, inputsize for fc layer
    '''
    def __init__(self, output_size, recur_model=None, kernelsize=2, stride=1):
        super().__init__(output_size, recur_model, kernelsize, stride)

        self.cnn2 = nn.Conv1d(BATCH_SIZE, BATCH_SIZE, kernel_size=kernelsize, stride=stride)
        cnn_out_size = (2 * self.rnn.hidden_size - kernelsize) // stride + 1
        cnn_out_size = (cnn_out_size - kernelsize) // stride + 1
        self.fc = nn.Linear(cnn_out_size, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.dropout(out[:, -1])
        out = self.cnn.forward(out)
        out = self.cnn2.forward(out)
        out = self.fc(out)
        return out


class Transphere_GRUModel_CNN_2layer_GRU(Transphere_GRUModel_CNN_2layer):
    '''
        model which fits a gru and fc layer after (gru layer - 2 cnn layers) with all frozen weights
        (not frozen: last gru and fc layer)
    '''
    def __init__(self, output_size, recur_model=None, hidden_size=302,
                 dropout=0, n_layers=2):
        super().__init__(output_size, recur_model, kernelsize=2, stride=1)
        # base=2-2
        kernelsize = 2
        for k in [self.cnn.parameters(), self.cnn2.parameters()]:
            for p in k:
                p.requires_grad = False
        cnn_out_size = (2 * self.rnn.hidden_size - kernelsize) + 1
        cnn_out_size = (cnn_out_size - kernelsize) + 1
        self.gru = nn.GRU(input_size=cnn_out_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          bidirectional=True, dropout=dropout,
                          batch_first=False)
        self.fc = nn.Linear(2 * hidden_size, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.dropout(out[:, -1])
        out = self.cnn.forward(out)
        out = self.cnn2.forward(out)
        out, _ = self.gru.forward(out)
        out = self.fc(out)
        return out


####################################################################################################
##       ABLATION STUDY MODELS                                                                    ##
####################################################################################################
## the following models are all variations of the Transphere_GRUModel_CNN_2layer_GRU              ##
##      with single layers left out                                                               ##
####################################################################################################


class Model_wo_first_GRU(Transphere_GRUModel_CNN_2layer_GRU):
    def __init__(self, output_size, recur_model=None, hidden_size=302,
                 dropout=0, n_layers=1):
        super().__init__(output_size, recur_model, hidden_size, dropout, n_layers)

        self.rnn = None     #first gru not used
        # base=2-2
        kernelsize = 2
        for k in [self.cnn.parameters(), self.cnn2.parameters()]:
            for p in k:
                p.requires_grad = True

        cnn_out_size = (Dataset.characters_count + 1 - kernelsize) + 1
        cnn_out_size = (cnn_out_size - kernelsize) + 1

        self.gru = nn.GRU(input_size=cnn_out_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          bidirectional=True, dropout=dropout,
                          batch_first=True)

        self.fc = nn.Linear(2 * hidden_size, self.output_size)

    def forward(self, batch):
        out = batch.permute(1, 0, 2)
        out = self.cnn.forward(out)
        out = self.cnn2.forward(out)
        out = out.permute(1, 0, 2)
        out, _ = self.gru.forward(out)
        out = self.dropout(out[:, -1])
        out = self.fc(out)
        return out


class Model_wo_CNN(Transphere_GRUModel_CNN_2layer_GRU):
    def __init__(self, output_size, recur_model=None, hidden_size=64,
                 dropout=0, n_layers=1):
        super().__init__(output_size, recur_model, hidden_size, dropout, n_layers)

        self.cnn = None
        self.cnn2 = None

        for k in [self.rnn.parameters()]: #first gru requires grad again
            for p in k:
                p.requires_grad = True

        self.gru = nn.GRU(input_size=2*self.rnn.hidden_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          bidirectional=True, dropout=dropout,
                          batch_first=False)

        self.fc = nn.Linear(2 * hidden_size, self.output_size)

    def forward(self, batch):
        out, _ = self.rnn(batch)
        out = self.dropout(out[:, -1])
        out, _ = self.gru.forward(out)
        out = self.fc(out)
        return out

#### w.o. second gru is not useful as it exists already as gru cnn-2_layer