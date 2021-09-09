from pickle import dump, load

import pytorch_lightning as pl
import torch
import sys

from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from TextDataset import TextDataset, PadSequence
from models.cnnlstm_model import LSTM_CNN_Model
from models.lstm_model import LSTM_Model
from torch.utils.data import DataLoader
from models.cnn_model import CNN_Model
import matplotlib.pyplot  as plt
import os
import numpy as np

if __name__ == '__main__':
    model_name = sys.argv[1]
    try:
        gpu_number = int(sys.argv[2])
    except:
        gpu_number = 0

    to_image = False
    if model_name.lower() == "lstm":
        try:
            hidden_size = int(sys.argv[3])
        except:
            hidden_size = 128
        model = LSTM_Model(hidden_size)
    elif model_name.lower() == "cnnlstm":
        model = LSTM_CNN_Model()
    elif model_name.lower() == "cnn":
        to_image = True
        model = CNN_Model()
    else:
        raise AttributeError("model name must be specified. 'lstm' / 'cnnlstm' / 'cnn'")

    train_dataset = TextDataset("ekman/training_ekman", to_image)
    validation_dataset = TextDataset("ekman/validation_ekman", to_image)
    test_dataset = TextDataset("ekman/test_ekman", to_image)

    train_dataloader = DataLoader(train_dataset, collate_fn=PadSequence(), batch_size=64, num_workers=24,
                                  shuffle=True)
    validation_daloader = DataLoader(validation_dataset, collate_fn=PadSequence(), batch_size=64, num_workers=24)
    test_daloader = DataLoader(test_dataset, collate_fn=PadSequence(), batch_size=64, num_workers=24)

    plot_pca(train_dataloader, 3)
    exit(0)

    es = pl.callbacks.EarlyStopping("validation_loss", patience=5)
    mc = pl.callbacks.ModelCheckpoint(monitor="validation_loss")
    logger = TensorBoardLogger(save_dir="lightning_logs", name=model_name.lower())
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=[gpu_number], precision=16, callbacks=[es, mc], logger=logger)
    else:
        trainer = pl.Trainer(logger=logger)

    trainer.fit(model, train_dataloader, validation_daloader)

    model = model.load_from_checkpoint(mc.best_model_path)
    trainer.test(model, test_daloader)
