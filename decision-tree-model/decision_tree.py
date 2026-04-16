# Decision Tree Classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Sample dataset (age, income → buy/not buy)
X = np.array([
    [22, 20000],
    [25, 25000],
    [47, 50000],
    [52, 60000],
    [46, 52000],
    [56, 65000]
])

y = [0, 0, 1, 1, 1, 1]  # 0 = No, 1 = Yes

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

print("Prediction:", pred)