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

all_evaluators = [ConfusionMatrix, AUROC, Accuracy, F1, Precision, Recall]


class Lightning_Super_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        for evaluator in all_evaluators:
            for step in ["train", "validation", "test"]:
                setattr(self, f"{step}_{evaluator.__name__}", evaluator(num_classes=2, compute_on_step=False))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-4)
        return opt

    def forward(self, x):
        raise NotImplementedError

    def loss(self, y, y_hat):
        return F.nll_loss(torch.log(y_hat), y.squeeze(), weight=torch.tensor([1., 10.], device=self.device))

    def log_evaluators(self, step_kind, y_hat, y):
        for e in all_evaluators:
            e_obj = getattr(self, f"{step_kind}_{e.__name__}")
            if "AUROC" in e.__name__:
                e_obj(y_hat.squeeze(), y.squeeze())
            else:
                e_obj(torch.argmax(y_hat, dim=1).flatten(), y.squeeze())
            if "ConfusionMatrix" not in e.__name__:
                self.log(f"{step_kind}_{e.__name__}", e_obj)

    def preform_step(self, kind, batch):
        x, lens, y = batch
        y = y.long()
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log(f"{kind}_loss", loss)
        self.log_evaluators(kind, y_hat, y)

        return loss

    def training_step(self, batch, batch_id):
        return self.preform_step("train", batch)

    def validation_step(self, batch, batch_id):
        return self.preform_step("validation", batch)

    def test_step(self, batch, batch_id):
        return self.preform_step("test", batch)

    def log_CM_helper(self, CM):
        df_cm = pd.DataFrame(CM, index=np.arange(2), columns=np.arange(2))
        plt.figure()
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        return im

    def preform_epoch_end(self, kind):
        tb = self.logger.experiment
        conf_mat_obj = getattr(self, f"{kind}_ConfusionMatrix")
        conf_mat = conf_mat_obj.compute().detach().cpu().numpy().astype(np.int)
        im = self.log_CM_helper(conf_mat)
        tb.add_image(f"{kind}_confusion_matrix", im, global_step=self.current_epoch)
        conf_mat_obj.reset()

    def training_epoch_end(self, outs):
        self.preform_epoch_end("train")

    def validation_epoch_end(self, outs):
        self.preform_epoch_end("validation")

    def test_epoch_end(self, outs):
        self.preform_epoch_end("test")
