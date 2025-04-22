import os
import numpy as np

def save_sequences(filename, sequences, variances):
    seqs = [s.tolist() for s in sequences]
    vars = np.array(variances)
    np.savez(filename, sequences=seqs, variances=vars)


def load_sequences(filename):
    if not os.path.exists(filename):
        return [], []
    data = np.load(filename, allow_pickle=True)
    return [np.array(seq) for seq in data['sequences']], data['variances']


def save_data(filename, sequences, variances):
    # Convert new data to list format
    new_sequences = [seq.tolist() for seq in sequences]
    new_variances = np.array(variances)

    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        existing_sequences = [seq.tolist() for seq in data["sequences"]]
        existing_variances = data["variances"]

        combined_sequences = existing_sequences + new_sequences
        combined_variances = np.concatenate((existing_variances, new_variances))
    else:
        combined_sequences = new_sequences
        combined_variances = new_variances

    np.savez(filename, sequences=combined_sequences, variances=combined_variances)

def load_data(filename):
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        sequences = [np.array(seq) for seq in data['sequences']]
        variances = data['variances']
        print(f"Loaded {len(sequences)} sequences from {filename}.")
        return sequences, variances
    else:
        print(f"No existing data found at {filename}. Starting fresh.")
        return [], []

def simulation_parameters(spots_per_m):
    from thermal_simulation.melting import melting_setup
    simulation_setup = melting_setup(spots_per_m)
    # Normalize the spot coordinates
    coords = simulation_setup['spot_coords']
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    normalized_coords = (coords - coords_min) / (coords_max - coords_min)
    simulation_setup['spot_coords'] = normalized_coords
    return simulation_setup

def combine_datasets(file_list, output_file):
    combined_sequences = []
    combined_variances = []
    for file in file_list:
        sequences, variances = load_data(file)
        if file == "./training_data/genetic_sequences.npz":
            sequences = sequences[0]
            variances = variances[0]
        combined_sequences.extend(sequences)
        combined_variances.extend(variances)
    save_data(output_file, combined_sequences, combined_variances)

def save_node_coords(simulation_setup, filename):
    coords = simulation_setup['spot_coords']
    np.save(filename, coords)
    print(f"Node coordinates saved to {filename}.")