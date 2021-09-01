import torch

from models.lightning_model import Lightning_Super_Model
from mlstmfcn import MLSTMfcn


class LSTM_CNN_Model(Lightning_Super_Model):
    def __init__(self):
        super().__init__()
        self.lstm = MLSTMfcn(num_classes=2, max_seq_len=None, num_features=7, num_lstm_layers=2, lstm_drop_p=0.6)

    def forward(self, x):
        x = self.lstm(x, torch.ones(x.shape[0]) * x.shape[1])
        return x
