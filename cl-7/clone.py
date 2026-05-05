import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# Convert labels:
# 0 = malignant → damage (1)
# 1 = benign → healthy (0)
y = np.where(y == 0, 1, 0)

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Step 2: Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: AIS Model
# -----------------------------
class AIS:
    def __init__(self, num_detectors=30, mutation_rate=0.05, clone_factor=3):
        self.num_detectors = num_detectors
        self.mutation_rate = mutation_rate
        self.clone_factor = clone_factor

    # Initialize detectors
    def initialize(self, X, y):
        indices = np.random.choice(len(X), self.num_detectors, replace=False)
        self.detectors = X[indices]
        self.detector_labels = y[indices]

    # Mutation function
    def mutate(self, detector):
        mutation = np.random.normal(0, self.mutation_rate, size=detector.shape)
        return np.clip(detector + mutation, 0, 1)

    # Affinity (fitness) calculation
    def affinity(self, detector, X, y):
        distances = np.linalg.norm(X - detector, axis=1)
        nearest_idx = np.argmin(distances)
        return 1 if y[nearest_idx] == self.get_label(detector) else 0

    # Get label of detector
    def get_label(self, detector):
        idx = np.where((self.detectors == detector).all(axis=1))[0][0]
        return self.detector_labels[idx]

    # Clone and mutate based on fitness
    def clone_and_mutate(self, X, y):
        new_detectors = []
        new_labels = []

        for i in range(len(self.detectors)):
            detector = self.detectors[i]
            label = self.detector_labels[i]

            fitness = self.affinity(detector, X, y)

            # More fit → more clones
            num_clones = self.clone_factor + fitness * 2

            for _ in range(num_clones):
                mutated = self.mutate(detector)
                new_detectors.append(mutated)
                new_labels.append(label)

        self.detectors = np.vstack((self.detectors, new_detectors))
        self.detector_labels = np.concatenate((self.detector_labels, new_labels))

    # Selection: keep best detectors
    def select_best(self, X, y):
        scores = []
        for i in range(len(self.detectors)):
            detector = self.detectors[i]
            label = self.detector_labels[i]

            distances = np.linalg.norm(X - detector, axis=1)
            nearest_idx = np.argmin(distances)

            score = 1 if y[nearest_idx] == label else 0
            scores.append(score)

        # Select top detectors
        indices = np.argsort(scores)[-self.num_detectors:]
        self.detectors = self.detectors[indices]
        self.detector_labels = self.detector_labels[indices]

    # Training with generations
    def fit(self, X, y, generations=5):
        self.initialize(X, y)

        for _ in range(generations):
            self.clone_and_mutate(X, y)
            self.select_best(X, y)

    # Prediction
    def predict(self, X):
        predictions = []
        for sample in X:
            distances = np.linalg.norm(self.detectors - sample, axis=1)
            closest = np.argmin(distances)
            predictions.append(self.detector_labels[closest])
        return np.array(predictions)

# -----------------------------
# Step 4: Train AIS
# -----------------------------
model = AIS(num_detectors=30, mutation_rate=0.05, clone_factor=3)
model.fit(X_train, y_train, generations=5)

# -----------------------------
# Step 5: Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Improved AIS Accuracy:", accuracy)
