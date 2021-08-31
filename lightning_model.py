import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sn
import pandas as pd
import numpy as np
import torchvision
import torch
import io

from torchmetrics import Accuracy, AUROC, ConfusionMatrix, F1, Recall, Precision
from PIL import Image


class Lightning_Super_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.CM_train = ConfusionMatrix(num_classes=2)
        self.CM_validation = ConfusionMatrix(num_classes=2)
        self.CM_test = ConfusionMatrix(num_classes=2)
        self.train_roc = AUROC(num_classes=2, compute_on_step=False)
        self.valid_roc = AUROC(num_classes=2, compute_on_step=False)
        self.test_roc = AUROC(num_classes=2, compute_on_step=False)

        self.train_evaluators = [Accuracy(num_classes=2, compute_on_step=False),
                                 F1(num_classes=2, compute_on_step=False),
                                 Precision(num_classes=2, compute_on_step=False),
                                 Recall(num_classes=2, compute_on_step=False)]
        self.validation_evaluators = [Accuracy(num_classes=2, compute_on_step=False),
                                      F1(num_classes=2, compute_on_step=False),
                                      Precision(num_classes=2, compute_on_step=False),
                                      Recall(num_classes=2, compute_on_step=False)]
        self.test_evaluators = [Accuracy(num_classes=2, compute_on_step=False),
                                F1(num_classes=2, compute_on_step=False),
                                Precision(num_classes=2, compute_on_step=False),
                                Recall(num_classes=2, compute_on_step=False)]

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-4)
        return opt

    def forward(self, x):
        raise NotImplementedError

    def loss(self, y, y_hat):
        return F.nll_loss(torch.log(y_hat), y.squeeze(), weight=torch.tensor([1., 10.], device=self.device))

    def log_evaluators(self, step_kind, evals, y_hat, y):
        for e in evals:
            e(y_hat, y)
            self.log(f"{step_kind}_{type(e)}", e)

    def training_step(self, batch, batch_id):
        x, lens, y = batch
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.train_roc(y_hat.squeeze(), y.squeeze())
        self.CM_train(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())

        self.log("train_loss", loss)
        self.log("train_roc", self.train_roc, on_step=True, on_epoch=True)

        self.log_evaluators("train", self.train_evaluators, torch.argmax(y_hat, dim=1).flatten(), y.squeeze())

        return loss

    def validation_step(self, batch, batch_id):
        x, lens, y = batch
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.CM_validation(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.valid_roc(y_hat.squeeze(), y.squeeze())

        self.log("validation_loss", loss)
        self.log("validation_roc", self.valid_roc, on_step=True, on_epoch=True)

        self.log_evaluators("validation", self.validation_evaluators, torch.argmax(y_hat, dim=1).flatten(), y.squeeze())

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
        im = self.log_CM_helper(conf_mat)
        tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)

    def training_epoch_end(self, outs):
        tb = self.logger.experiment
        # confusion matrix
        conf_mat = self.CM_train.compute().detach().cpu().numpy().astype(np.int)
        self.CM_train.reset()
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
        self.CM_test(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
        self.test_roc(y_hat.squeeze(), y.squeeze())

        self.log("test_loss", loss)
        self.log("test_roc", self.test_roc, on_step=True, on_epoch=True)

        self.log_evaluators("test", self.test_evaluators, torch.argmax(y_hat, dim=1).flatten(), y.squeeze())

        return loss
