import torch
from torch.utils.data import Dataset, DataLoader
from thermal_simulation.melting import melting_setup, compute_melting
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
import os

def generate_training_data(
    simulation_setup, num_sequences=800, save_filename="training_data.npz", log_interval=1
):
    internal_nodes = simulation_setup["internal_nodes"]

    # Check if the training data file exists
    if os.path.exists(save_filename):
        # Load existing data
        existing_data = np.load(save_filename, allow_pickle=True)
        sequences = existing_data["sequences"].tolist()
        variances = existing_data["variances"].tolist()
    else:
        sequences = []
        variances = []

    for i in range(num_sequences):
        # Generate a random melting sequence
        sequence = np.random.permutation(internal_nodes)
        # Compute the variance metric
        if i == 0:
            variance_metric, variance_list = compute_melting(
                simulation_setup, melting_sequence=sequence, log_interval=log_interval
            )
            print(f"Variance metric for first sequence: {variance_metric}")
        else:
            variance_metric, _ = compute_melting(
                simulation_setup, melting_sequence=sequence, log_interval=log_interval
            )
        # Add to training data
        sequences.append(sequence)
        variances.append(variance_metric)

    # Convert sequences and variances to NumPy arrays
    sequences_array = np.array(sequences, dtype=object)
    variances_array = np.array(variances)

    # Save the combined data back to the file
    np.savez(save_filename, sequences=sequences_array, variances=variances_array)

    # Prepare data for the dataset
    training_data = list(zip(sequences_array, variances_array))
    return training_data

def load_training_data(filename="training_data.npz"):
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        sequences = data["sequences"]
        variances = data["variances"]
        training_data = list(zip(sequences, variances))
        print(f"Loaded {len(training_data)} training examples from '{filename}'.")
        return training_data
    else:
        print(f"No existing training data found at '{filename}'.")
        return []

def simulation_parameters(pixels_per_m):
    simulation_setup = melting_setup(pixels_per_m)
    
    # Normalize the pixel coordinates
    coords = simulation_setup['pixel_coords']
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    normalized_coords = (coords - coords_min) / (coords_max - coords_min)
    simulation_setup['pixel_coords'] = normalized_coords

    return simulation_setup

class MeltingSequenceDataset(Dataset):
    def __init__(self, training_data, simulation_setup):
        self.sequences = [
            torch.tensor(seq.tolist(), dtype=torch.long) for seq, _ in training_data
        ]
        self.targets = torch.tensor(
            [metric for _, metric in training_data], dtype=torch.float
        )
        self.node_coords = simulation_setup["pixel_coords"]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        coords = torch.tensor(self.node_coords[sequence.numpy()], dtype=torch.float)
        return sequence, coords, target

def collate_fn(batch):
    sequences, coords_list, targets = zip(*batch)
    # Pad sequences and coordinates to the same length
    seq_lengths = [len(seq) for seq in sequences]
    max_length = max(seq_lengths)

    padded_sequences = [
        torch.cat([seq, torch.zeros(max_length - len(seq), dtype=torch.long)])
        for seq in sequences
    ]
    padded_coords = [
        torch.cat([coords, torch.zeros(max_length - len(coords), coords.size(1))])
        for coords in coords_list
    ]

    padded_sequences = torch.stack(padded_sequences).transpose(0, 1)  # Shape: [max_length, batch_size]
    padded_coords = torch.stack(padded_coords).transpose(0, 1)  # Shape: [max_length, batch_size, coord_dim]
    targets = torch.stack(targets)
    return padded_sequences, padded_coords, targets

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.6, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_length, batch_size, embedding_dim]
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)

class MeltSequenceTransformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        coord_dim,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super(MeltSequenceTransformer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.coord_projection = nn.Linear(coord_dim, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward=256, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(embedding_dim, 1)

    def forward(self, src, coords):
        # src: [seq_length, batch_size]
        # coords: [seq_length, batch_size, coord_dim]
        src_embed = self.embedding(src)  # [seq_length, batch_size, embedding_dim]
        coords_embed = self.coord_projection(coords)  # [seq_length, batch_size, embedding_dim]
        combined = src_embed + coords_embed  # Combine embeddings
        combined = combined * math.sqrt(self.embedding.embedding_dim)
        combined = self.pos_encoder(combined)
        output = self.transformer_encoder(combined)
        output = output.mean(dim=0)
        output = self.fc_out(output)
        return output.squeeze()

def train_transformer_model(dataloader, num_nodes, coord_dim, num_epochs=10, learning_rate=0.1):
    model = MeltSequenceTransformer(num_nodes=num_nodes, coord_dim=coord_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    epoch_losses = []  # List to store average loss per epoch

    for epoch in range(num_epochs):
        total_loss = 0
        for sequences_batch, coords_batch, targets_batch in dataloader:
            optimizer.zero_grad()

            outputs = model(sequences_batch, coords_batch)

            loss = criterion(outputs, targets_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences_batch.size(1)

        avg_loss = total_loss / len(dataloader.dataset)
        epoch_losses.append(avg_loss)  # Record average loss for this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "trained_model.pth")
    return model, epoch_losses

def find_best_sequence(model, simulation_setup, num_attempts=1000):
    model.eval()
    internal_nodes = simulation_setup["internal_nodes"]
    num_nodes = len(internal_nodes)
    node_coords = simulation_setup["pixel_coords"]
    coord_dim = node_coords.shape[1]

    best_sequence = None
    lowest_predicted_variance = float("inf")

    for _ in range(num_attempts):
        sequence = np.random.permutation(internal_nodes)
        sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(1)  # Shape: [seq_length, 1]
        coords = torch.tensor(node_coords[sequence], dtype=torch.float).unsqueeze(1)  # Shape: [seq_length, 1, coord_dim]

        with torch.no_grad():
            predicted_variance = model(sequence_tensor, coords)
            if isinstance(predicted_variance, torch.Tensor):
                predicted_variance = predicted_variance.item()
            else:
                predicted_variance = float(predicted_variance)

        if predicted_variance < lowest_predicted_variance:
            lowest_predicted_variance = predicted_variance
            best_sequence = sequence.copy()

    return best_sequence, lowest_predicted_variance

def training_main(pixels_per_m, generate_new_data=True, num_sequences=800):
    training_data_file = "training_data" + str(pixels_per_m) + ".npz"
    simulation_setup = simulation_parameters(pixels_per_m)
    internal_nodes = simulation_setup["internal_nodes"]
    num_nodes = max(internal_nodes) + 1
    coord_dim = simulation_setup['pixel_coords'].shape[1]

    training_data = load_training_data(training_data_file)

    if generate_new_data:
        new_training_data = generate_training_data(
            simulation_setup,
            num_sequences=num_sequences,
            save_filename=training_data_file
        )
        training_data.extend(new_training_data)

    dataset = MeltingSequenceDataset(training_data, simulation_setup)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Train the model and get epoch losses
    model, epoch_losses = train_transformer_model(
        dataloader, num_nodes=num_nodes, coord_dim=coord_dim, num_epochs=10, learning_rate=0.1
    )

    # Plot the loss curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.show()

    best_sequence, predicted_variance = find_best_sequence(
        model, simulation_setup, num_attempts=100000
    )
    print(f"Best predicted variance: {predicted_variance}")

    # Compute variance metric and variance list
    actual_variance, variance_list = compute_melting(
        simulation_setup, melting_sequence=best_sequence, plot_melting=True, log_interval=1
    )
    print(f"Actual variance of the best sequence: {actual_variance}")

    # Plot the variance over time
    plt.figure()
    plt.plot(range(len(variance_list)), variance_list)
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    plt.title('Variance Over Time')
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    training_main(pixels_per_m=600, generate_new_data=True, num_sequences=10)
