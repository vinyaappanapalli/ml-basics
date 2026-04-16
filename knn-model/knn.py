# KNN Classification Example

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset (height, weight → class)
X = np.array([
    [150, 50],
    [160, 60],
    [170, 70],
    [180, 80],
    [155, 55],
    [165, 65]
])

y = [0, 0, 1, 1, 0, 1]  # 0 = Class A, 1 = Class B

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

print("Prediction:", prediction)