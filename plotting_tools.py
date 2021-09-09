import matplotlib.pyplot  as plt
import numpy as np
import torch
import os

from sklearn.decomposition import IncrementalPCA
from pickle import dump, load
from tqdm import tqdm


def plot_pca(dataloader, dims=2):
    if dims != 2 and dims != 3:
        raise AttributeError("only 2 or 3 dims are supported")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d' if dims == 3 else None)

    if f"pca_train_model_{dims}d.pkl" in os.listdir():
        with open(f"pca_train_model_{dims}d.pkl", "rb") as f:
            # f = open(n, "wb")
            pca = load(f)
        # f.close()
    else:
        pca = IncrementalPCA(n_components=dims)
        print("Creating pca")
        for B in tqdm(dataloader):
            x, len, y = B
            x[x == 0] = torch.tensor(float("nan"))
            pca.partial_fit(np.nanmean(x, axis=1))
        with open(f"pca_train_model_{dims}d.pkl", "wb") as f:
            dump(pca, f)

    print("Implemanting creating pca")
    for i, B in enumerate(tqdm(dataloader)):
        x, len, y = B
        x[x == 0] = torch.tensor(float("nan"))
        d = pca.transform(np.nanmean(x, axis=1))
        trues = d[(y == 1).squeeze().numpy()]
        falses = d[(y == 0).squeeze().numpy()]
        ax.scatter(*[falses[:, j] for j in range(dims)], s=1, c="r", label="Negative" if i == 0 else "", alpha=0.7)
        ax.scatter(*[trues[:, j] for j in range(dims)], s=1, c="b", label="Positive" if i == 0 else "", alpha=0.7)

    plt.legend()
    plt.savefig(f"pca_{dims}.png")
    plt.show()
