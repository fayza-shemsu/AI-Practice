import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    "Price": [100000, 150000, 200000, 250000, 300000, 400000],
    "Size": [800, 1000, 1200, 1500, 1800, 2200],
    "Bedrooms": [2, 3, 3, 4, 4, 5],
    "Age": [20, 15, 10, 8, 5, 2]
}

df = pd.DataFrame(data)

# Step 2: Features (X) and Target (y)
X = df[["Size", "Bedrooms", "Age"]]
y = df["Price"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create model
model = LinearRegression()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
preds = model.predict(X_test)

# Step 7: Show results
print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(preds)
