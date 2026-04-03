# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
data = pd.read_csv("data.csv")
data.columns = data.columns.str.strip()
print(f"DEBUG: Column names after stripping are: {list(data.columns)}")

# Show first 5 rows
print(data.head())

# Handle missing values
data = data.dropna()


# Change column names based on your dataset
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", metrics.r2_score(y_test, y_pred))

# Plot results
# Add line for perfect prediction reference
plt.plot(y_test, y_test, color='red')
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")
plt.show()
