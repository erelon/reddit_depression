from pickle import _Unpickler
import json
from struct import unpack

import torch
from torch.utils.data import DataLoader, IterableDataset
from sklearn.svm import SVC
import pytorch_lightning as pl
from tqdm import tqdm

from lstm_model import LSTM_Model


def _crawl_on_pickle(fd):
    s = fd.tell()
    load_proto = fd.read(1)
    proto = fd.read(1)
    load_frame = fd.read(1)
    if load_proto == b'':
        raise EOFError
    assert (load_frame == b'\x95') and (proto == b'\x04') and (load_proto == b'\x80')
    frame = unpack('<Q', fd.read(8))[0]
    frame += 11
    fd.seek(frame + s)
    while fd.read(1) == b'\x95':
        frame += unpack('<Q', fd.read(8))[0] + 8 + 1
        fd.seek(frame + s)

    return frame + s


def crawl_on_pickle(fd):
    places = [0]
    while True:
        try:
            fr = _crawl_on_pickle(fd)
            fd.seek(fr)
            places.append(fr)
        except EOFError:
            break
    fd.seek(0)
    return places[:-1]

#
# if __name__ == '__main__':
#     with open("training_ekman", 'rb') as f:
#         places = crawl_on_pickle(f)
#         un = _Unpickler(f)
#         for i in tqdm(places):
#             f.seek(i)
#             un.load()

