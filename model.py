import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv("data/train.csv")  # Adjust path if your CSV is in 'data/' folder

# Fill missing values (like Tuesday)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    # For simplicity, let's use numeric columns only
X = df[numeric_cols].drop("SalePrice", axis=1)  # Features
y = df["SalePrice"]  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Evaluate using Mean Squared Error
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error on Test Set: {mse}")