# Install DEAP (only if not installed)
# !pip install deap

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

# -------------------------------
# Fitness Function (maximize function)
# -------------------------------
def eval_func(individual):
    x, y = individual
    return (-(x**2 + y**2) + 10,)   # Max value = 10 at (0,0)

# -------------------------------
# Avoid redefinition errors
# -------------------------------
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# -------------------------------
# Toolbox Setup
# -------------------------------
toolbox = base.Toolbox()

# Attribute generator (-5 to 5)
toolbox.register("attr_float", random.uniform, -5, 5)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic Operators
toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# -------------------------------
# Main Genetic Algorithm
# -------------------------------
def run_ga():
    population = toolbox.population(n=50)
    generations = 30

    fitness_history = []

    for gen in range(generations):

        # Evaluate fitness
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        # Select individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for i in range(0, len(offspring), 2):
            if random.random() < 0.7:
                toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Recalculate fitness for new individuals
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Replace population
        population[:] = offspring

        # Store best fitness
        best = tools.selBest(population, 1)[0]
        fitness_history.append(best.fitness.values[0])

        print(f"Generation {gen}: Best Fitness = {best.fitness.values[0]}")

    return population, fitness_history

# -------------------------------
# Run Algorithm
# -------------------------------
final_pop, fitness_history = run_ga()

# -------------------------------
# Plot Results
# -------------------------------
plt.plot(fitness_history)
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness over Generations")
plt.show()

# -------------------------------
# Best Solution
# -------------------------------
best_ind = tools.selBest(final_pop, 1)[0]
print("\nBest Individual:", best_ind)
print("Best Fitness:", best_ind.fitness.values[0])