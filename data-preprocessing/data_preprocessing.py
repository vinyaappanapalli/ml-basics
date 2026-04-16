# Data Preprocessing Example

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sample dataset
data = {
    "Name": ["A", "B", "C", "D"],
    "Age": [25, None, 30, 35],
    "City": ["NY", "LA", "NY", "SF"],
    "Salary": [50000, 60000, 55000, 65000]
}

df = pd.DataFrame(data)

# 1. Handle missing values
df["Age"].fillna(df["Age"].mean(), inplace=True)

# 2. Encoding categorical data
le = LabelEncoder()
df["City"] = le.fit_transform(df["City"])

# 3. Feature scaling
scaler = StandardScaler()
df["Salary"] = scaler.fit_transform(df[["Salary"]])

print(df)