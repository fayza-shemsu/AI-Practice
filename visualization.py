# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Load the data
# -----------------------------
df = pd.read_csv("data/train.csv")  # make sure the path matches your folder
print("Data Loaded Successfully\n")

# -----------------------------
# Step 2: Fill missing numeric values
# -----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# -----------------------------
# Step 3: Correlation Heatmap
# -----------------------------
numeric_df = df[numeric_cols]  # numeric only
corr = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlations (Numeric Only)")
plt.show()

# Optional: focus on SalePrice correlation
plt.figure(figsize=(8, 6))
sns.heatmap(corr[["SalePrice"]].sort_values(by="SalePrice", ascending=False),
            annot=True, cmap="coolwarm")
plt.title("Features Most Correlated with SalePrice")
plt.show()

# -----------------------------
# Step 4: Distribution of SalePrice
# -----------------------------
plt.figure(figsize=(8, 6))
plt.hist(df["SalePrice"], bins=30, color="skyblue", edgecolor="black")
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# Step 5: Boxplot to Identify Outliers
# -----------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(x=df["SalePrice"])
plt.title("Boxplot of SalePrice (Outliers)")
plt.show()

# Optional: Boxplot for OverallQual vs SalePrice
plt.figure(figsize=(10, 6))
sns.boxplot(x="OverallQual", y="SalePrice", data=df)
plt.title("SalePrice by Overall Quality")
plt.show()