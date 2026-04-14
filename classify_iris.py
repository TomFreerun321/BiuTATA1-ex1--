"""
================================================================================
Iris Dataset Classification — MLP Neural Network (4-Class Edition)
Course: Machine Learning
Assignment: Task 7
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, mean_squared_error
)

# Set visual style
sns.set_theme(style="whitegrid")

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA ACQUISITION & SYNTHESIS
# ──────────────────────────────────────────────────────────────────────────────
print("Step 1: Preparing 4-class dataset...")

iris = load_iris()
X = iris.data
y = iris.target
original_names = list(iris.target_names)
feature_names = [f.replace(' (cm)', '') for f in iris.feature_names]

# Synthesizing 4th species: "Aviv Iris"
# Defined with slightly larger proportions to be distinct
np.random.seed(42)
aviv_X = np.zeros((50, 4))
aviv_X[:, 0] = np.random.normal(7.8, 0.4, 50)  # Sepal Length
aviv_X[:, 1] = np.random.normal(3.8, 0.3, 50)  # Sepal Width
aviv_X[:, 2] = np.random.normal(6.9, 0.5, 50)  # Petal Length
aviv_X[:, 3] = np.random.normal(2.5, 0.2, 50)  # Petal Width

X_ext = np.vstack([X, aviv_X])
y_ext = np.concatenate([y, np.full(50, 3)])
class_names = original_names + ['aviv iris']

# Export CSV
df = pd.DataFrame(X_ext, columns=iris.feature_names)
df['species'] = [class_names[i] for i in y_ext]
df.to_csv('iris_dataset.csv', index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
print("Step 2: Preprocessing and Scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X_ext, y_ext, test_size=0.20, random_state=42, stratify=y_ext
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────────────────────────────────────
print("Step 3: Training MLP Model...")
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=600,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ──────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION (Adding MSE as per instructions)
# ──────────────────────────────────────────────────────────────────────────────
print("Step 4: Comprehensive Evaluation...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Calculate MSE (Treating labels as numeric for this metric as per assignment example)
mse = mean_squared_error(y_test, y_pred)

print(f"✓ Accuracy: {accuracy * 100:.1f}%")
print(f"✓ MSE: {mse:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: 4x4 Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='magma', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'4x4 Confusion Matrix\nAccuracy: {accuracy*100:.1f}% | MSE: {mse:.3f}', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=200)
plt.close()

# Plot 2: Convergence (Loss Curve)
plt.figure(figsize=(9, 5))
plt.plot(model.loss_curve_, color='#8B5CF6', linewidth=2.5)
plt.title('Training Convergence (Loss Curve)', fontsize=14, fontweight='bold')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=200)
plt.close()

# Plot 3: Feature Distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = sns.color_palette("viridis", 4)
for i, (feat, ax) in enumerate(zip(feature_names, axes.flatten())):
    for j, (cls, col) in enumerate(zip(class_names, colors)):
        sns.kdeplot(X_ext[y_ext == j, i], fill=True, ax=ax, color=col, label=cls, alpha=0.3)
    ax.set_title(f'Distribution: {feat}', fontweight='bold')
    ax.legend()
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=200)
plt.close()

print("\n" + "="*50)
print("  ALL FILES PERFECTED AND SAVED")
print("="*50)
