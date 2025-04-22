import numpy as np
from thermal_simulation.melting import MeltingSimulator
from .utils import save_sequences

def generate_random_sequences(simulation_setup, num_sequences, filename):
    internal_nodes = simulation_setup["internal_nodes"]
    simulator = MeltingSimulator(simulation_setup=simulation_setup)

    sequences, variances = [], []
    for i in range(num_sequences):
        seq = np.random.permutation(internal_nodes)
        simulator.simulation_setup["melting_sequence"] = seq
        metric, _ = simulator.compute_melting(plot_melting=False)
        sequences.append(seq)
        variances.append(metric)
        #if i == 0:
        #    print(f"Variance metric for first sequence: {metric}")

    save_sequences(filename, sequences, variances)
    return sequences, variances