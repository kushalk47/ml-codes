import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Olivetti Faces Dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes Classifier
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Cross-validation Accuracy: {cross_val_score(model, X, y, cv=5).mean() * 100:.2f}%")

# Display some test images with predictions
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, true, pred in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{true}, P:{pred}")
    ax.axis('off')
plt.show()
