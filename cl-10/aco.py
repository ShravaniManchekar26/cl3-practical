# ============================================
# ANT COLONY OPTIMIZATION FOR TSP
# ============================================

import numpy as np
import random

# ------------------ DISTANCE MATRIX ------------------
distances = np.array([
    [0, 2, 2, 5, 7],
    [2, 0, 4, 8, 2],
    [2, 4, 0, 1, 3],
    [5, 8, 1, 0, 2],
    [7, 2, 3, 2, 0]
])

n_cities = len(distances)
n_ants = 5
n_iterations = 50

alpha = 1      # pheromone importance
beta = 2       # heuristic importance
evaporation = 0.5

# ------------------ INITIALIZATION ------------------
pheromone = np.ones((n_cities, n_cities))
heuristic = 1 / (distances + 1e-10)

# ------------------ FUNCTION TO CALCULATE DISTANCE ------------------
def calculate_distance(path):
    total = 0
    for i in range(len(path) - 1):
        total += distances[path[i]][path[i+1]]
    total += distances[path[-1]][path[0]]  # return to start
    return total

# ------------------ ACO ALGORITHM ------------------
best_path = None
best_distance = float('inf')

for iteration in range(n_iterations):
    all_paths = []

    for ant in range(n_ants):
        visited = [random.randint(0, n_cities - 1)]

        while len(visited) < n_cities:
            current = visited[-1]
            probabilities = []

            for city in range(n_cities):
                if city not in visited:
                    prob = (pheromone[current][city] ** alpha) * (heuristic[current][city] ** beta)
                    probabilities.append((city, prob))

            cities, probs = zip(*probabilities)
            probs = np.array(probs)
            probs = probs / probs.sum()

            next_city = np.random.choice(cities, p=probs)
            visited.append(next_city)

        all_paths.append(visited)

    # ------------------ PHEROMONE UPDATE ------------------
    pheromone *= (1 - evaporation)

    for path in all_paths:
        dist = calculate_distance(path)

        for i in range(len(path) - 1):
            pheromone[path[i]][path[i+1]] += 1 / dist

        if dist < best_distance:
            best_distance = dist
            best_path = path

# ------------------ OUTPUT ------------------
print("Best Path:", best_path)
print("Best Distance:", best_distance)