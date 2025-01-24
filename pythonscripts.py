# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Logistic Regression ---
# Standardize the features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train_scaled, y_train)

# Predict and evaluate Logistic Regression
y_pred_logreg = logreg.predict(X_test_scaled)
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.2f}")
print(classification_report(y_test, y_pred_logreg, target_names=iris.target_names))

# --- Decision Tree ---
# Train the Decision Tree model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predict and evaluate Decision Tree
y_pred_tree = tree.predict(X_test)
print("\nDecision Tree Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))

# --- Cross-Validation ---
logreg_scores = cross_val_score(logreg, X, y, cv=10)  # 10-fold cross-validation
tree_scores = cross_val_score(tree, X, y, cv=10)  # 10-fold cross-validation

print(f"\nLogistic Regression Cross-Validation Accuracy: {logreg_scores.mean():.2f} (+/- {logreg_scores.std():.2f})")
print(f"Decision Tree Cross-Validation Accuracy: {tree_scores.mean():.2f} (+/- {tree_scores.std():.2f})")

# --- Adding Noise ---
# Add random noise to features
noise = np.random.normal(0, 0.1, X.shape)  # Mean 0, std 0.1
X_noisy = X + noise

# Train and evaluate models with noisy data
logreg.fit(X_noisy, y)
y_pred_logreg_noisy = logreg.predict(X_noisy)
print("\nLogistic Regression Accuracy on Noisy Data:")
print(f"Accuracy: {accuracy_score(y, y_pred_logreg_noisy):.2f}")

tree.fit(X_noisy, y)
y_pred_tree_noisy = tree.predict(X_noisy)
print("Decision Tree Accuracy on Noisy Data:")
print(f"Accuracy: {accuracy_score(y, y_pred_tree_noisy):.2f}")

# --- Visualize Decision Boundaries ---
# Take the first two features for 2D visualization
X_vis = X[:, :2]

# Create a mesh grid for plotting decision boundaries
h = .02
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# --- Logistic Regression Decision Boundaries ---
logreg.fit(X_vis, y)
Z_logreg = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z_logreg = Z_logreg.reshape(xx.shape)

# --- Decision Tree Decision Boundaries ---
tree.fit(X_vis, y)
Z_tree = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z_tree = Z_tree.reshape(xx.shape)

# Plot Logistic Regression boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_logreg, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o')
plt.title("Logistic Regression")

# Plot Decision Tree boundaries
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_tree, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', marker='o')
plt.title("Decision Tree")

plt.show()
