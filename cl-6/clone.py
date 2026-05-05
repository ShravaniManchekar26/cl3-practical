import random
import matplotlib.pyplot as plt

# -------------------------------
# 1. Fitness Function (Minimization → Maximization)
# -------------------------------
def fitness(x):
    return -x ** 2   # maximize this → minimizes x²

# -------------------------------
# 2. Initialize Population
# -------------------------------
def init_population(size, lower, upper):
    return [random.uniform(lower, upper) for _ in range(size)]

# -------------------------------
# 3. Mutation (Adaptive)
# -------------------------------
def mutate(x, lower, upper, mutation_rate):
    mutation_strength = (upper - lower) * mutation_rate
    x = x + random.uniform(-mutation_strength, mutation_strength)
    return max(lower, min(x, upper))

# -------------------------------
# 4. Clone and Mutate
# -------------------------------
def clone_and_mutate(population, clone_factor, lower, upper):
    clones = []
    max_fit = max(fitness(x) for x in population)

    for x in population:
        # Avoid division by zero
        if max_fit == 0:
            num_clones = clone_factor
        else:
            num_clones = int(clone_factor * (fitness(x) / max_fit))

        num_clones = max(1, num_clones)  # at least 1 clone

        for _ in range(num_clones):
            mutation_rate = 1 / (abs(fitness(x)) + 1)  # better → less mutation
            mutated = mutate(x, lower, upper, mutation_rate)
            clones.append(mutated)

    return clones

# -------------------------------
# 5. Selection (Best Individuals)
# -------------------------------
def select_best(population, size):
    return sorted(population, key=fitness, reverse=True)[:size]

# -------------------------------
# 6. Add Random Cells (Diversity)
# -------------------------------
def add_random(population, lower, upper, num_random):
    for _ in range(num_random):
        population.append(random.uniform(lower, upper))
    return population

# -------------------------------
# 7. Parameters
# -------------------------------
POP_SIZE = 5
GENERATIONS = 10
CLONE_FACTOR = 3
LOWER, UPPER = -10, 10

# -------------------------------
# 8. Run Algorithm
# -------------------------------
population = init_population(POP_SIZE, LOWER, UPPER)
best_fitness_history = []

for gen in range(GENERATIONS):
    clones = clone_and_mutate(population, CLONE_FACTOR, LOWER, UPPER)

    combined = population + clones

    # Add diversity
    combined = add_random(combined, LOWER, UPPER, 2)

    # Select best antibodies
    population = select_best(combined, POP_SIZE)

    best = population[0]
    best_fitness_history.append(fitness(best))

    print(f"Generation {gen+1}, Best Solution: {best:.4f}, Fitness: {fitness(best):.4f}")

# -------------------------------
# 9. Final Result
# -------------------------------
print("\nFinal Best Solution:", population[0])
print("Final Fitness:", fitness(population[0]))

# -------------------------------
# 10. Plot
# -------------------------------
plt.plot(best_fitness_history, marker='o')
plt.title("Fitness vs Generations (Clonal Selection Algorithm)")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid()
plt.show()
