import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, genetic_generator):
        self.ge = genetic_generator
        self.ga = genetic_generator.ga

    def plot_metrics(self):
        generations = range(len(self.ge.losses[0]))
        num_sequences = len(self.ge.losses)
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        for i in range(num_sequences):
            plt.plot(generations, self.ge.losses[i], label = f"Sequence {i + 1}")
            print(self.ge.losses[i])
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Loss Over Generations')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        for i in range(num_sequences):
            plt.plot(generations, self.ge.mutation_rates[i], label = f"Sequence {i + 1}")
        plt.xlabel('Generation')
        plt.ylabel('Mutation Rate')
        plt.title('Mutation Rate Over Generations')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        for i in range(num_sequences):
            plt.plot(generations, self.ge.diversities[i], label = f"Sequence {i + 1}")
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.title('Population Diversity Over Generations')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def plot_loss_over_generations(self):
        generations = range(len(self.ge.loss_history))
        plt.plot(generations, self.ge.loss_history)
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Loss Over Generations')
        plt.show()

    def plot_mutation_rate_over_generations(self):
        generations = range(len(self.ge.mutation_rate_history))
        plt.plot(generations, self.ge.mutation_rate_history)
        plt.xlabel('Generation')
        plt.ylabel('Mutation Rate')
        plt.title('Mutation Rate Over Generations')
        plt.show()

    def plot_population_diversity(self):
        generations = range(len(self.ge.diversity_history))
        plt.plot(generations, self.ge.diversity_history)
        plt.xlabel('Generation')
        plt.ylabel('Diversity Metric')
        plt.title('Population Diversity Over Generations')
        plt.show()
