1️⃣ Distributions

Definition:
A distribution shows how the values of a variable are spread across its range. It helps us understand patterns, trends, and anomalies.

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Price'], bins=10, kde=True)  # Histogram + density
plt.show()

sns.boxplot(x=df['Price'])  # Detects outliers visually
plt.show()

 2️⃣ Correlations

Definition:
Correlation measures how two variables move together:

Positive correlation → both increase together

Negative correlation → one increases, the other decreases

Zero correlation → no linear relationship

sns.scatterplot(x='Bedrooms', y='Price', data=df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

3️⃣ Outliers

Definition:
Outliers are values far from the typical range of a variable. They can be errors, rare events, or opportunities.