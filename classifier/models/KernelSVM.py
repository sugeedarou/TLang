import numpy as np
import torch
from ..settings import BATCH_SIZE

class MultiCatKernelSVM(torch.nn.Module):
    '''
      Used to compute a Kernalized SVM for multi class outputs

      requires x.shape[0] = n_samples
    '''

    def __init__(self, input_size, output_size, kernel='rbf',
                 gamma_init=1.0, train_gamma=True):
        super().__init__()
        assert kernel in ['linear', 'rbf']
        self._output_size = output_size

        if kernel == 'linear':
            self._kernel = self.linear
            self._num_c = 2
        elif kernel == 'rbf':
            self._kernel = self.rbf
            self._num_c = input_size
            self._gamma = torch.nn.Parameter(torch.FloatTensor([gamma_init]),
                                             requires_grad=train_gamma)
        else:
            assert False
        self._w = torch.nn.Linear(in_features=BATCH_SIZE,
                                  out_features=output_size)

    def rbf(self, x):
        '''
        requires x.shape = [n_samples, n_input_dim]
        '''
        y = x.repeat(x.size(0), 1, 1)
        return torch.exp(-self._gamma * ((x[:, np.newaxis] - y) ** 2).sum(dim=-1))

    @staticmethod
    def linear(x):
        return x

    def forward(self, x):
        '''
          computes n_cat independend kernelized svms
            in dim [n_samples, n_input_dim]
            out dim [n_cats]
        '''
        y = self._kernel(x)
        y = self._w(y)
        return y