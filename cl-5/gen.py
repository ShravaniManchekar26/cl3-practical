import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset shape:", X.shape)

# -------------------------------
# STEP 2: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 3: Proper Scaling (IMPORTANT FIX)
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# STEP 4: Activation Function
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------------------
# STEP 5: Neural Network (Single Layer)
# -------------------------------
def neural_network(X, weights):
    w = weights[:-1]   # weights
    b = weights[-1]    # bias
    return sigmoid(np.dot(X, w) + b)

# -------------------------------
# STEP 6: Fitness Function (Log Loss)
# -------------------------------
def fitness(weights):
    preds = neural_network(X_train, weights)
    
    # Avoid log(0)
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    
    loss = -np.mean(
        y_train * np.log(preds) + (1 - y_train) * np.log(1 - preds)
    )
    return loss

# -------------------------------
# STEP 7: GA PARAMETERS
# -------------------------------
POP_SIZE = 20
GENERATIONS = 30
MUTATION_RATE = 0.3   # slightly reduced for stability
DIM = X_train.shape[1] + 1  # weights + bias

# -------------------------------
# STEP 8: Initialize Population
# -------------------------------
def init_population():
    return [np.random.randn(DIM) for _ in range(POP_SIZE)]

# -------------------------------
# STEP 9: Selection (Best Half)
# -------------------------------
def selection(pop, scores):
    indices = np.argsort(scores)
    return [pop[i] for i in indices[:POP_SIZE//2]]

# -------------------------------
# STEP 10: Crossover
# -------------------------------
def crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    return np.concatenate((p1[:point], p2[point:]))

# -------------------------------
# STEP 11: Mutation
# -------------------------------
def mutate(child):
    for i in range(len(child)):
        if random.random() < MUTATION_RATE:
            child[i] += np.random.normal(0, 0.3)
    return child

# -------------------------------
# STEP 12: GA TRAINING LOOP
# -------------------------------
population = init_population()
errors = []

for gen in range(GENERATIONS):
    scores = [fitness(ind) for ind in population]
    
    best_error = min(scores)
    errors.append(best_error)
    
    print(f"Generation {gen+1}, Best Error: {best_error:.4f}")
    
    parents = selection(population, scores)
    new_population = parents.copy()
    
    while len(new_population) < POP_SIZE:
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)
    
    population = new_population

# -------------------------------
# STEP 13: Best Solution
# -------------------------------
final_scores = [fitness(ind) for ind in population]
best_idx = np.argmin(final_scores)
best_weights = population[best_idx]

print("\nOptimized Weights Found")

# -------------------------------
# STEP 14: Prediction
# -------------------------------
def predict(X, weights):
    probs = neural_network(X, weights)
    return (probs > 0.5).astype(int)

y_pred = predict(X_test, best_weights)

accuracy = np.mean(y_pred == y_test)
print("Test Accuracy:", accuracy)

# -------------------------------
# STEP 15: Plot Error Graph
# -------------------------------
plt.plot(errors, marker='o')
plt.title("Error vs Generations")
plt.xlabel("Generation")
plt.ylabel("Error (Loss)")
plt.grid()
plt.show()