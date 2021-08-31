import pickle
from pickle import _Unpickler
import json
from struct import unpack
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from sklearn.svm import SVC
import pytorch_lightning as pl
import os
from tqdm import tqdm

from crawl_on_pickle import crawl_on_pickle
from lstm_model import LSTM_Model


class TextDataset(Dataset):
    def __init__(self, path: str):
        self.pfin = path
        self.reinit = False

        path = path.split("/")
        if (path[-1] + "_indexes.pkl") not in os.listdir("/".join(path[:-1])):
            print("creating indexes")
            f = open(self.pfin, "rb")
            self.p_indexes = crawl_on_pickle(f)
            ff = open("/".join(path) + "_indexes.pkl", "wb")
            pickle.dump(self.p_indexes, ff)
            ff.close()
            f.close()
            ff = open("/".join(path) + "_indexes.pkl")
        else:
            print("loding indexes")
            ff = open("/".join(path) + "_indexes.pkl", "rb")
            self.p_indexes = pickle.load(ff)
            ff.close()

    def __len__(self):
        return len(self.p_indexes)

    def __getitem__(self, index):
        if self.reinit is False:
            self.reinit = True
            self.f = open(self.pfin, "rb")
            self.un = _Unpickler(self.f)
        return self.yield_from_pickle(self.p_indexes[index])

    def __del__(self):
        try:
            self.f.close()
        except:
            pass

    def yield_from_pickle(self, seek_to):
        self.f.seek(seek_to)
        line = self.un.load()

        posts_t = torch.stack([torch.tensor(i[1]) for i in line["posts"]])
        if line["label"] == "control":
            label_t = torch.zeros(1)
        else:
            label_t = torch.ones(1)
        return [posts_t, label_t]


class PadSequence:
    def __call__(self, batch):
        # Let's assume that each element in "batch" is a tuple (data, label).
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        # Get each sequence and pad it
        sequences = [x[0] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        # Also need to store the length of each sequence
        # This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])

        # Don't forget to grab the labels of the *sorted* batch
        labels = torch.stack([x[1] for x in sorted_batch])

        return sequences_padded, lengths, labels


if __name__ == '__main__':
    train_dataset = TextDataset("ekman/training_ekman")
    validation_dataset = TextDataset("ekman/validation_ekman")
    test_dataset = TextDataset("ekman/test_ekman")

    train_dataloader = DataLoader(train_dataset, collate_fn=PadSequence(), batch_size=64, num_workers=24, shuffle=True)
    validation_daloader = DataLoader(validation_dataset, collate_fn=PadSequence(), batch_size=64, num_workers=24)
    test_daloader = DataLoader(test_dataset, collate_fn=PadSequence(), batch_size=64, num_workers=24)

    es = pl.callbacks.EarlyStopping("validation_loss", patience=5)
    mc = pl.callbacks.ModelCheckpoint(monitor="validation_loss")
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs")
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=[0], precision=16, callbacks=[es, mc], logger=logger)
    else:
        trainer = pl.Trainer(logger=logger)

    model = LSTM_Model()
    trainer.fit(model, train_dataloader, validation_daloader)

    model = model.load_from_checkpoint(mc.best_model_path)
    trainer.test(model, test_daloader)
