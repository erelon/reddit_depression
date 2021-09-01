import pickle
import torch
import os

from crawl_on_pickle import crawl_on_pickle
from torch.utils.data import Dataset
from pickle import _Unpickler


class TextDataset(Dataset):
    def __init__(self, path: str, to_image: bool = False):
        self.pfin = path
        self.to_image = to_image
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
            # ff = open("/".join(path) + "_indexes.pkl")
        else:
            print("loading indexes")
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

        if self.to_image:
            pass

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
