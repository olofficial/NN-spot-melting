import numpy as np
from numpy.random import permutation, rand, choice
from joblib import Parallel, delayed
from thermal_simulation.melting import SimulationSetup, MeltingSimulator

class GeneticAlgorithm:
    def __init__(self, spots_per_m, generations=100, checkpoint_interval=10, mutation_rate=0.2, elitism=True, n_jobs=-1):
        self.spots_per_m = spots_per_m
        self.generations = generations
        self.checkpoint_interval = checkpoint_interval
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.n_jobs = n_jobs
        self.loss_history = []
        self.mutation_rate_history = []
        self.diversity_history = []

    def initialize_population(self, genome_space, pop_size):
        return [permutation(genome_space) for _ in range(pop_size)]

    def evaluate_fitness(self, population):
        return Parallel(n_jobs=self.n_jobs)(delayed(self.fitness)(ind) for ind in population)

    def fitness(self, sequence):
        setup = SimulationSetup(self.spots_per_m, melting_sequence=sequence).simulation_setup
        sim = MeltingSimulator(setup)
        value, _ = sim.compute_melting(plot_melting=False)
        return 1.0 / (value + 1e-6)

    def selection(self, pop, fitness, k=2):
        idx = choice(len(pop), size=k, replace=False)
        best = idx[np.argmax([fitness[i] for i in idx])]
        return pop[best]

    def crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(choice(range(size), 2, replace=False))
        child = np.full(size, -1)
        child[a:b+1] = p1[a:b+1]
        pos = 0
        for gene in p2:
            if gene not in child:
                while child[pos] != -1:
                    pos += 1
                child[pos] = gene
        return child

    def mutate(self, seq):
        if rand() < self.mutation_rate:
            i, j = choice(len(seq), 2, replace=False)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    def run(self):
        setup = SimulationSetup(self.spots_per_m).simulation_setup
        nodes = setup["internal_nodes"]
        pop = self.initialize_population(nodes, len(nodes))
        fit = np.array(self.evaluate_fitness(pop))

        for gen in range(self.generations):
            self.mutation_rate = 2 * np.exp(-((gen / 200) + 1))
            new_pop = []

            if self.elitism:
                elite_n = max(1, int(0.1 * len(pop)))
                elite_idx = np.argsort(fit)[-elite_n:]
                new_pop.extend([pop[i].copy() for i in elite_idx])

            while len(new_pop) < len(pop):
                p1 = self.selection(pop, fit)
                p2 = self.selection(pop, fit)
                c = self.mutate(self.crossover(p1, p2))
                new_pop.append(c)

            pop = new_pop
            fit = np.array(self.evaluate_fitness(pop))

            loss = np.mean(1.0 / (fit + 1e-6))
            self.loss_history.append(loss)
            self.mutation_rate_history.append(self.mutation_rate)
            self.diversity_history.append(self.population_diversity(pop))

        best_idx = np.argmax(fit)
        return pop[best_idx], 1.0 / (fit[best_idx] + 1e-6)

    def population_diversity(self, pop):
        n = len(pop)
        if n < 2: return 0.0
        dist = sum(np.sum(p1 != p2) for i, p1 in enumerate(pop) for p2 in pop[i+1:])
        return dist / (len(pop[0]) * n * (n - 1) / 2)