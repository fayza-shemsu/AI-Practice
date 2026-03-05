import pandas as pd

# Load data
df = pd.read_csv("house_prices.csv")

# Check missing values
print(df.isnull().sum())

# Fill missing numerical values
df["Price"] = df["Price"].fillna(df["Price"].mean())
df["Size"] = df["Size"].fillna(df["Size"].mean())

# Fill missing categorical values
df["Location"] = df["Location"].fillna(df["Location"].mode()[0])

# Filter houses with more than 3 bedrooms
filtered_df = df[df["Bedrooms"] > 3]

print(filtered_df.head())