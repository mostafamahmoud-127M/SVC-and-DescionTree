import matplotlib
matplotlib.use('TkAgg')  # Forces Matplotlib to use a standalone window
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
svc = SVC()
tree = DecisionTreeClassifier()

# Train the classifiers
svc.fit(X_train, y_train)
tree.fit(X_train, y_train)

# Predictions
svc_preds = svc.predict(X_test)
tree_preds = tree.predict(X_test)

# Evaluate models
svc_accuracy = accuracy_score(y_test, svc_preds)
tree_accuracy = accuracy_score(y_test, tree_preds)

svc_report = classification_report(y_test, svc_preds)
tree_report = classification_report(y_test, tree_preds)

# Save results to a text file
with open("result.txt", "w") as f:
    f.write("=== Support Vector Classifier ===\n")
    f.write(f"Accuracy: {svc_accuracy:.4f}\n")
    f.write(svc_report + "\n")
    f.write("=== Decision Tree Classifier ===\n")
    f.write(f"Accuracy: {tree_accuracy:.4f}\n")
    f.write(tree_report + "\n")

print("Results saved to result.txt")

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SVC Confusion Matrix
svc_cm = confusion_matrix(y_test, svc_preds)
sns.heatmap(svc_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('SVC Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# Decision Tree Confusion Matrix
tree_cm = confusion_matrix(y_test, tree_preds)
sns.heatmap(tree_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Decision Tree Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()
