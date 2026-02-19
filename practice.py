
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Check current directory
print("Working directory:", os.getcwd())

# Load the House Prices data
df = pd.read_csv('data/train.csv')

# Check first few rows
print("\nFirst 5 rows:")
print(df.head())

# Check for nulls
print("\nMissing values:")
print(df.isnull().sum())

# ✅ FILL MISSING VALUES
print("\n--- Filling Missing Values ---")
print("LotFrontage missing before:", df['LotFrontage'].isnull().sum())
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
print("LotFrontage missing after:", df['LotFrontage'].isnull().sum())

# Filter houses with > 3 bedrooms
big_houses = df[df['BedroomAbvGr'] > 3]
print(f"\nHouses with 3+ bedrooms: {len(big_houses)}")

# ✅ BONUS: More filtering practice
print("\n--- Bonus Practice ---")
affordable_big = df[(df['BedroomAbvGr'] > 3) & (df['SalePrice'] < 300000)]
print(f"Affordable big houses (< $300k): {len(affordable_big)}")

pool_houses = df[df['PoolArea'] > 0]
print(f"Houses with pools: {len(pool_houses)}")
print(f"Avg price with pool: ${pool_houses['SalePrice'].mean():,.2f}")
# ✅ DATA VISUALIZATION SECTION
import matplotlib.pyplot as plt

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')

# 1. Price Distribution Histogram
plt.figure(figsize=(10, 6))
df['SalePrice'].hist(bins=50, edgecolor='black')
plt.title('House Price Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Sale Price ($)')
plt.ylabel('Number of Houses')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Average Price by Bedroom Count
plt.figure(figsize=(8, 5))
df.groupby('BedroomAbvGr')['SalePrice'].mean().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Price by Bedroom Count', fontsize=14, fontweight='bold')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Sale Price ($)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. LotFrontage vs SalePrice (Scatter Plot)
plt.figure(figsize=(10, 6))
plt.scatter(df['LotFrontage'], df['SalePrice'], alpha=0.5, s=20)
plt.title('Lot Frontage vs Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Lot Frontage (feet)')
plt.ylabel('Sale Price ($)')
plt.tight_layout()
plt.show()
# 4. Correlation Heatmap
plt.figure(figsize=(12, 8))

correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(correlation_matrix, cmap='coolwarm')

plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
# ================================
# 5️⃣ Supervised Learning Section
# ================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Select Features (X) and Target (y)
# Let's use some numeric features for simplicity
features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'TotRmsAbvGrd', 'BedroomAbvGr']
X = df[features]
y = df['SalePrice']

print("\nFeatures (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

# 2️⃣ Split Data into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 3️⃣ Initialize Linear Regression Model
model = LinearRegression()

# 4️⃣ Train the Model
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# 5️⃣ Make Predictions
preds = model.predict(X_test)

# 6️⃣ Evaluate Model
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\nMean Squared Error (MSE): {mse:,.2f}")
print(f"R2 Score: {r2:.2f}")

# 7️⃣ Optional: Compare Actual vs Predicted (first 10)
comparison = pd.DataFrame({"Actual": y_test[:10], "Predicted": preds[:10]})
print("\nActual vs Predicted (first 10 samples):")
print(comparison)


