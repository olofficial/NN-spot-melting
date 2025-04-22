import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from model import MeltSequenceTransformer
from dataset import load_training_data, normalize_coords, MeltingSequenceDataset


def train_model(dataloader, num_nodes, coord_dim, epochs=10, lr=0.001):
    model = MeltSequenceTransformer(num_nodes, coord_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    history = []
    for epoch in range(epochs):
        epoch_loss = 0
        for seq, coords, target in dataloader:
            optimizer.zero_grad()
            pred = model(seq, coords)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * seq.size(1)
        avg = epoch_loss / len(dataloader.dataset)
        history.append(avg)
        print(f"Epoch {epoch+1}: Loss = {avg:.4f}")

    torch.save(model.state_dict(), "trained_model.pth")
    return model, history


def find_best_sequence(model, coords, internal_nodes, attempts=10000):
    model.eval()
    best_seq, best_score = None, float('inf')
    with torch.no_grad():
        for _ in range(attempts):
            seq = np.random.permutation(internal_nodes)
            seq_t = torch.tensor(seq, dtype=torch.long).unsqueeze(1)
            coords_t = torch.tensor(coords[seq], dtype=torch.float).unsqueeze(1)
            pred = model(seq_t, coords_t).item()
            if pred < best_score:
                best_score = pred
                best_seq = seq.copy()
    return best_seq, best_score


def run(spots_per_m, path):
    from thermal_simulation.melting import SimulationSetup

    setup = SimulationSetup(spots_per_m).simulation_setup
    coords = normalize_coords(setup['spot_coords'])
    training_data = load_training_data(path)
    dataset = MeltingSequenceDataset(training_data, coords)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_nodes = max(setup['internal_nodes']) + 1
    coord_dim = coords.shape[1]

    model, losses = train_model(dataloader, num_nodes, coord_dim)

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    best_seq, best_pred = find_best_sequence(model, coords, setup['internal_nodes'])
    print(f"Best predicted variance: {best_pred:.4f}")
    return best_seq, best_pred
