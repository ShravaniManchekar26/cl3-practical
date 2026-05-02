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

# Normalize data (important for distance calculation)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Step 2: Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
# Step 3: AIS Model
# -----------------------------
class AIS:
    def __init__(self, num_detectors=30, mutation_rate=0.05):
        self.num_detectors = num_detectors
        self.mutation_rate = mutation_rate

    # Train model (initialize detectors)
    def train(self, X, y):
        indices = np.random.choice(len(X), self.num_detectors, replace=False)
        self.detectors = X[indices]
        self.detector_labels = y[indices]

    # Mutation (learning improvement)
    def mutate(self, detector):
        mutation = np.random.normal(0, self.mutation_rate, size=detector.shape)
        return np.clip(detector + mutation, 0, 1)

    # Clonal selection (optional improvement)
    def clone_and_mutate(self):
        new_detectors = []
        new_labels = []

        for i in range(len(self.detectors)):
            detector = self.detectors[i]
            label = self.detector_labels[i]

            # Create clones
            for _ in range(2):  # 2 clones per detector
                mutated = self.mutate(detector)
                new_detectors.append(mutated)
                new_labels.append(label)

        # Combine original + clones
        self.detectors = np.vstack((self.detectors, new_detectors))
        self.detector_labels = np.concatenate((self.detector_labels, new_labels))

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
model = AIS(num_detectors=30, mutation_rate=0.05)
model.train(X_train, y_train)

# Apply clonal selection (important AIS step)
model.clone_and_mutate()

# -----------------------------
# Step 5: Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("AIS Accuracy:", accuracy)