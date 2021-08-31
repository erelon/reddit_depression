import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sn
import pandas as pd
import numpy as np
import torchvision
import torch
import io

from torchmetrics import Accuracy, AUROC, ConfusionMatrix
from lightning_model import Lightning_Super_Model
from PIL import Image


class LSTM_Model(Lightning_Super_Model):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.2)
        self.lin = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1, :]
        x = self.lin(x)
        return torch.softmax(x, dim=1)
