import torch

from models.lightning_model import Lightning_Super_Model
from torchvision.models import vgg11_bn


class CNN_Model(Lightning_Super_Model):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Conv2d(1, 3, (1, 1))
        self.vgg = vgg11_bn(pretrained=True)
        self.vgg.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.vgg(x)
        return torch.softmax(x, dim=1)
