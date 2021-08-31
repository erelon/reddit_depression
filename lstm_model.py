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
from PIL import Image


class LSTM_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_acc = Accuracy(num_classes=2)
        self.valid_acc = Accuracy(num_classes=2)
        self.test_acc = Accuracy(num_classes=2)
        self.CM_train = ConfusionMatrix(num_classes=2)
        self.CM_validation = ConfusionMatrix(num_classes=2)
        self.CM_test = ConfusionMatrix(num_classes=2)
        self.train_roc = AUROC(num_classes=2, compute_on_step=False)
        self.valid_roc = AUROC(num_classes=2, compute_on_step=False)
        self.test_roc = AUROC(num_classes=2, compute_on_step=False)
        hidden_size = 128
        self.lstm = torch.nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.2)
        self.lin = torch.nn.Linear(hidden_size, 2)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-4)
        return opt

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = x[:, -1, :]
        x = self.lin(x)
        return torch.softmax(x, dim=1)

    def loss(self, y, y_hat):
        return F.nll_loss(torch.log(y_hat), y.squeeze(), weight=torch.tensor([1., 10.], device=self.device))

    def training_step(self, batch, batch_id):
        x, lens, y = batch
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss)
        self.train_acc(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.CM_train(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.train_roc(y_hat.squeeze(), y.squeeze())
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_roc", self.train_roc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_id):
        x, lens, y = batch
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("validation_loss", loss)
        self.valid_acc(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.CM_validation(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.valid_roc(y_hat.squeeze(), y.squeeze())
        self.log("validation_acc", self.valid_acc, on_step=True, on_epoch=True)
        self.log("validation_roc", self.valid_roc, on_step=True, on_epoch=True)

        return loss

    def log_CM_helper(self, CM):
        df_cm = pd.DataFrame(
            CM,
            index=np.arange(2),
            columns=np.arange(2))
        plt.figure()
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        return im

    def validation_epoch_end(self, outs):
        tb = self.logger.experiment
        # confusion matrix
        conf_mat = self.CM_validation.compute().detach().cpu().numpy().astype(np.int)
        self.CM_validation.reset()
        # self.valid_roc.reset()
        # self.valid_acc.reset()
        im = self.log_CM_helper(conf_mat)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

    def training_epoch_end(self, outs):
        tb = self.logger.experiment
        # confusion matrix
        conf_mat = self.CM_train.compute().detach().cpu().numpy().astype(np.int)
        self.CM_train.reset()
        # self.train_roc.reset()
        # self.train_acc.reset()
        im = self.log_CM_helper(conf_mat)
        tb.add_image("train_confusion_matrix", im, global_step=self.current_epoch)

    def test_epoch_end(self, outs):
        tb = self.logger.experiment
        # confusion matrix
        conf_mat = self.CM_test.compute().detach().cpu().numpy().astype(np.int)
        im = self.log_CM_helper(conf_mat)
        tb.add_image("test_confusion_matrix", im, global_step=self.current_epoch)

    def test_step(self, batch, batch_id):
        x, lens, y = batch
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("test_loss", loss)
        self.test_acc(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.CM_test(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.test_roc(y_hat.squeeze(), y.squeeze())
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True)
        self.log("test_roc", self.test_roc, on_step=True, on_epoch=True)

        return loss
