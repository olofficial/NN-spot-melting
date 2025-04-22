from .genetic_algorithm import GeneticAlgorithm
from .utils import save_sequences

def generate_genetic_sequences(simulation_setup, num_sequences, filename, generations=10):
    sequences, variances = [], []
    losses, mutation_rates, diversities = [], [], []

    for _ in range(num_sequences):
        ga = GeneticAlgorithm(
            spots_per_m=simulation_setup['spots_per_m'],
            generations=generations,
            checkpoint_interval=10,
            mutation_rate=0.8,
            elitism=True,
            n_jobs=-1
        )
        best_seq, best_val = ga.run()
        sequences.append(best_seq)
        variances.append(best_val)
        losses.append(ga.loss_history)
        mutation_rates.append(ga.mutation_rate_history)
        diversities.append(ga.diversity_history)

    save_sequences(filename, sequences, variances)
    return sequences, variances, losses, mutation_rates, diversities