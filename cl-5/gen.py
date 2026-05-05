import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# STEP 1: Simulated Spray Drying Dataset
# -------------------------------
# Inputs: Temperature, Airflow, Feed Rate
# Output: Powder Yield (continuous)

np.random.seed(42)
X = np.random.uniform(50, 200, (200, 3))  # 3 features
y = (0.3 * X[:, 0] + 0.5 * X[:, 1] - 0.2 * X[:, 2] +
     np.random.normal(0, 5, 200))  # regression output

# -------------------------------
# STEP 2: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 3: Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# STEP 4: Neural Network
# -------------------------------
def neural_network(X, weights):
    w = weights[:-1]
    b = weights[-1]
    return np.dot(X, w) + b   # Linear (Regression)

# -------------------------------
# STEP 5: Fitness Function (MSE)
# -------------------------------
def fitness(weights):
    preds = neural_network(X_train, weights)
    return np.mean((y_train - preds) ** 2)

# -------------------------------
# STEP 6: GA PARAMETER RANGES (Optimized)
# -------------------------------
POP_RANGE = (10, 50)
MUT_RANGE = (0.1, 0.5)
GEN_RANGE = (20, 60)

# Randomly choose GA parameters (Optimization concept)
POP_SIZE = random.randint(*POP_RANGE)
MUTATION_RATE = random.uniform(*MUT_RANGE)
GENERATIONS = random.randint(*GEN_RANGE)

DIM = X_train.shape[1] + 1

print("Optimized GA Parameters:")
print("Population:", POP_SIZE)
print("Mutation Rate:", MUTATION_RATE)
print("Generations:", GENERATIONS)

# -------------------------------
# STEP 7: Initialize Population
# -------------------------------
def init_population():
    return [np.random.randn(DIM) for _ in range(POP_SIZE)]

# -------------------------------
# STEP 8: Selection
# -------------------------------
def selection(pop, scores):
    idx = np.argsort(scores)
    return [pop[i] for i in idx[:POP_SIZE // 2]]

# -------------------------------
# STEP 9: Crossover
# -------------------------------
def crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    return np.concatenate((p1[:point], p2[point:]))

# -------------------------------
# STEP 10: Mutation
# -------------------------------
def mutate(child):
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] += np.random.normal(0, 0.5)
    return child

# -------------------------------
# STEP 11: GA Training
# -------------------------------
population = init_population()
errors = []

for gen in range(GENERATIONS):
    scores = [fitness(ind) for ind in population]
    best_error = min(scores)
    errors.append(best_error)

    print(f"Generation {gen+1}, Best MSE: {best_error:.4f}")

    parents = selection(population, scores)
    new_population = parents.copy()

    while len(new_population) < POP_SIZE:
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

# -------------------------------
# STEP 12: Best Solution
# -------------------------------
final_scores = [fitness(ind) for ind in population]
best_idx = np.argmin(final_scores)
best_weights = population[best_idx]

print("\nOptimized Weights Found")

# -------------------------------
# STEP 13: Prediction
# -------------------------------
y_pred = neural_network(X_test, best_weights)

# Evaluation (MSE)
test_mse = np.mean((y_test - y_pred) ** 2)
print("Test MSE:", test_mse)

# -------------------------------
# STEP 14: Plot Error Graph
# -------------------------------
plt.plot(errors, marker='o')
plt.title("MSE vs Generations")
plt.xlabel("Generation")
plt.ylabel("Error (MSE)")
plt.grid()
plt.show()
