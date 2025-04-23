import torch
from torch.utils.data import Dataset
import numpy as np

class MeltingSequenceDataset(Dataset):
    def __init__(self, training_data, coords):
        self.sequences = [torch.tensor(seq, dtype=torch.long) for seq, _ in training_data]
        self.targets = torch.tensor([v for _, v in training_data], dtype=torch.float)
        self.coords = coords

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        coord = torch.tensor(self.coords[seq], dtype=torch.float)
        return seq, coord, target


def load_training_data(filename):
    data = np.load(filename, allow_pickle=True)
    sequences = data['sequences']
    variances = data['variances']
    return list(zip(sequences, variances))


def normalize_coords(coords):
    cmin, cmax = coords.min(axis=0), coords.max(axis=0)
    return (coords - cmin) / (cmax - cmin)


def plot_variance_metric(log_variances):
    import matplotlib.pyplot as plt
    steps = range(len(log_variances))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, log_variances)
    plt.xlabel("Simulation Step")
    plt.ylabel("log(Variance)")
    plt.title("Log Variance of Internal Temperature Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_temperature_distribution(coords, temperatures):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial import cKDTree

    x, y = coords[:, 0], coords[:, 1]
    grid_res = 300
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_coords = np.stack([Xi.ravel(), Yi.ravel()], axis=-1)

    tree = cKDTree(coords)
    dist, idx = tree.query(grid_coords)
    grid_T = temperatures[idx].reshape(grid_res, grid_res)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='hot', aspect='equal')
    plt.colorbar(label='Temperature')
    plt.title("2D Heat Distribution")
    plt.tight_layout()
    plt.show()