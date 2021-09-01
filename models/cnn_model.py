import torch

from models.lightning_model import Lightning_Super_Model
from torchvision.models import vgg11_bn


class CNN_Model(Lightning_Super_Model):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Conv2d(1, 3, (1, 1))
        self.vgg = vgg11_bn(pretrained=True)

    def forward(self, x):
        x = self.cnn(x)
        x = self.vgg(x)
        return x
