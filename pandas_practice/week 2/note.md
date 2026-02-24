1️⃣ What is Feature Engineering?

Definition:
Feature engineering is the process of transforming raw data into meaningful input features that improve the performance of machine learning models.

Features = columns/attributes in your dataset that a model uses to learn patterns.

Good features can make a huge difference in model accuracy.

📘  Feature Engineering & Categorical Data

Machines only understand numbers

ML algorithms cannot work directly with text like "Red" or "Downtown".

Categorical text must be converted into numeric features.

Common issues with categorical data

Text labels: "Red", "Blue", "Green" cannot be interpreted directly.

High cardinality: Categories with many unique values can cause sparse matrices.

Ordinal vs Nominal:

Ordinal: categories have an order (e.g., "Low" < "Medium" < "High")

Nominal: categories have no order (e.g., "Red", "Blue", "Green")

Numeric features need scaling

Numeric columns like Salary and Age may have different ranges.

Scaling helps algorithms converge faster and treat features equally.