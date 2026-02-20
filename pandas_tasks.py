import pandas as pd

# Load dataset
df = pd.read_csv("data/train.csv")

print("Data Loaded Successfully")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill numeric columns with mean
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check missing after cleaning (PRINT OUTSIDE LOOP)
print("\nMissing After Cleaning:")
print(df.isnull().sum().sum())

# Filter houses with > 3 bedrooms
big_houses = df[df["BedroomAbvGr"] > 3]

print("\nHouses with more than 3 bedrooms:")
print(len(big_houses))