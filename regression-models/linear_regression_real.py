# Regression Model: Linear Regression (Real Dataset)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset (hours studied vs marks)
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 60, 65, 70, 80, 85]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Predictions:", pred)