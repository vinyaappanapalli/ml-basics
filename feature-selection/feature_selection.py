# Feature Selection using Correlation

import pandas as pd

# Sample dataset
data = {
    "Age": [25, 30, 35, 40, 45],
    "Salary": [50000, 60000, 70000, 80000, 90000],
    "Experience": [1, 3, 5, 7, 9],
    "Purchased": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Correlation matrix
corr = df.corr()

print("Correlation Matrix:\n", corr)

# Selecting features with high correlation to target
target_corr = corr["Purchased"].abs()

important_features = target_corr[target_corr > 0.5]

print("\nImportant Features:\n", important_features)