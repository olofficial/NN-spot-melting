from data_generation.random_data_generator import generate_random_sequences
from data_generation.genetic_data_generator import generate_genetic_sequences
from data_generation.utils import save_node_coords, combine_datasets
from thermal_simulation.melting import SimulationSetup


def generate_training_data(total_sequences, random_ratio, spots_per_m, visualize=False):
    sim_setup = SimulationSetup(spots_per_m).simulation_setup
    save_node_coords(sim_setup, 'training_data/node_coords.npy')

    n_rand = int(total_sequences * random_ratio)
    n_gen = total_sequences - n_rand

    random_sequences, random_variances = generate_random_sequences(sim_setup, n_rand, 'training_data/random_sequences.npz')
    genetic_sequences, genetic_variances, losses, mutation_rates, diversities = generate_genetic_sequences(sim_setup, n_gen, 'training_data/genetic_sequences.npz')

    combine_datasets([
        'training_data/random_sequences.npz',
        'training_data/genetic_sequences.npz'
    ], 'training_data/combined_dataset.npz')

    #if visualize:
    #    for i, logv in enumerate(losses):
    #        print(f"GA run {i+1} loss trajectory:")
    #        plot_variance_metric(logv)

    print("Training data generation completed.")


if __name__ == "__main__":
    generate_training_data(total_sequences=1, random_ratio=0, spots_per_m=600, visualize=True)
