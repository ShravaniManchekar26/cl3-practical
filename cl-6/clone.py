import random
import matplotlib.pyplot as plt

# Fitness function
def fitness(x):
    return x ** 2

# Initialize population
def init_population(size, lower, upper):
    return [random.uniform(lower, upper) for _ in range(size)]

# Mutation with probability
def mutate(x, lower, upper, mutation_rate):
    if random.random() < mutation_rate:
        x = x + random.uniform(-1, 1)
    return max(lower, min(x, upper))

# Clone based on fitness
def clone_and_mutate(population, clone_factor, lower, upper):
    clones = []
    max_fit = max(fitness(x) for x in population)

    for x in population:
        # Better fitness → more clones
        num_clones = int(clone_factor * (fitness(x) / max_fit + 0.1))
        
        for _ in range(num_clones):
            mutation_rate = 1 / (fitness(x) + 1)  # inverse relation
            mutated = mutate(x, lower, upper, mutation_rate)
            clones.append(mutated)

    return clones

# Selection
def select_best(population, size):
    return sorted(population, key=fitness, reverse=True)[:size]

# Add random individuals (diversity)
def add_random(population, lower, upper, num_random):
    for _ in range(num_random):
        population.append(random.uniform(lower, upper))
    return population


# Parameters
POP_SIZE = 5
GENERATIONS = 10
CLONE_FACTOR = 3
LOWER, UPPER = -10, 10

# Run algorithm
population = init_population(POP_SIZE, LOWER, UPPER)
best_fitness_history = []

for gen in range(GENERATIONS):
    clones = clone_and_mutate(population, CLONE_FACTOR, LOWER, UPPER)

    combined = population + clones

    # Add diversity
    combined = add_random(combined, LOWER, UPPER, 2)

    population = select_best(combined, POP_SIZE)

    best = population[0]
    best_fitness_history.append(fitness(best))

    print(f"Generation {gen+1}, Best Solution: {best:.4f}, Fitness: {fitness(best):.4f}")

# Final result
print("\nFinal Best Solution:", population[0])
print("Final Fitness:", fitness(population[0]))

# Plot
plt.plot(best_fitness_history, marker='o')
plt.title("Fitness vs Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.grid()
plt.show()