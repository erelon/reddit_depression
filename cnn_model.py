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
from torchvision.models import vgg11_bn
from PIL import Image


class CNN_Model(Lightning_Super_Model):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Conv2d(1, 3, (1, 1))
        self.vgg = vgg11_bn(pretrained=True)

    def forward(self, x):
        x = self.cnn(x)
        x = self.vgg(x)
        return x
