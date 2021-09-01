import torch

from models.lightning_model import Lightning_Super_Model


class LSTM_Model(Lightning_Super_Model):
    def __init__(self):
        super().__init__()
        hidden_size = 128
        self.lstm = torch.nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.2)
        self.lin = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1, :]
        x = self.lin(x)
        return torch.softmax(x, dim=1)
