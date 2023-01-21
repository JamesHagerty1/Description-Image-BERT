import torch
import torch.nn as nn


################################################################################


class BERT(nn.Module):
    def __init__(self, c):
        super(BERT, self).__init__()

    def forward(self, x, y_i):
        print("forward")
        print(x.shape, y_i.shape)
        return -1