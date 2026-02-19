Day 04 – Supervised Learning
1️⃣ Supervised Learning

Definition:
Supervised learning is a type of machine learning where the model is trained on a dataset that includes both input features and known outputs (targets). The goal is for the model to learn the mapping from input to output so it can predict unseen data.

Key Points:

The algorithm learns from labeled data.

It can be used for regression (predicting numbers) or classification (predicting categories).

Example: Predicting house prices based on features like size, bedrooms, and location.

Size (sqft)	Bedrooms	Price ($)
1200	3	300000
1500	4	400000
800	2	200000

Here, Size and Bedrooms are features, and Price is the target.
 **Regression:** Predict continuous values (e.g., house price)
- **Classification:** Predict categories (e.g., spam/not spam)

**Example:** Predict house prices based on size, bedrooms, etc.


2️⃣ Features (X) vs Target (y)

Features (X): The input variables used to predict something.

Target (y): The output variable we want to predict.

Example in code:

import pandas as pd

# Example dataset
data = {
    "Size": [1200, 1500, 800],
    "Bedrooms": [3, 4, 2],
    "Price": [300000, 400000, 200000]
}

df = pd.DataFrame(data)

# Features
X = df[["Size", "Bedrooms"]]

# Target
y = df["Price"]

print("Features (X):")
print(X)
print("\nTarget (y):")
print(y)


Output:

Features (X):
   Size  Bedrooms
0  1200        3
1  1500        4
2   800        2

Target (y):
0    300000
1    400000
2    200000
Name: Price, dtype: int64

3️⃣ Train/Test Split

Definition:
To evaluate a model’s performance on unseen data, we split the dataset into:

Training set: Used to train the model

Testing set: Used to evaluate the model

Common split: 80% training, 20% testing

Code Example:

from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:")
print(X_train)
print("\nX_test:")
print(X_test)


Key Points:

test_size=0.2 → 20% of data is for testing

random_state=42 → ensures results are reproducible
Linear Regression Model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

5. Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MSE: {mse}, R2: {r2}")


MSE: Measures average squared error (lower is better)

R²: Proportion of variance explained (closer to 1 is better)

6. Output Example
MSE: 1929813681.84
R2: 0.75

Actual vs Predicted (first 10 samples):
      Actual      Predicted
892   154500  173031.53
1105  325000  290224.23
...

✅ Notes

First Linear Regression model completed!

Shows that even a simple model can explain a large portion of house price variance.
✅ Summary
Concept	Meaning
Supervised Learning	Learn from labeled data to predict output
Features (X)	Input variables used to make predictions
Target (y)	Output variable we want to predict
Train/Test Split	Split data into training and testing sets to evaluate model