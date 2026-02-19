📘 Theory: Core Concepts

1️⃣ Distributions
A distribution shows how values of a variable are spread.
It tells us:

Are most values small or large?
Is the data symmetric?
Is it skewed?
Is it normal (bell-shaped)?

🔎 Example:

If you plot house prices:

If most houses cost 100k–200k and few cost 1M,
the distribution is right-skewed.

If values are evenly spread, distribution is balanced.

Why it matters:
Many ML models assume data is normally distributed.
Skewed data may require transformation (like log scaling).

2️⃣ Correlations

Correlation measures the relationship between two variables.

It answers:

When one variable increases, does the other increase?

Do they move in opposite directions?Or are they unrelated?

Correlation values range from:

+1 → strong positive relationship

0 → no relationship

-1 → strong negative relationship

🔎 Example:

If:

House size increases → Price increases
→ strong positive correlation

If:

Age of house increases → Price decreases
→ negative correlation

Why it matters:
Highly correlated features with Price are important predictors.
Highly correlated features with each other may cause multicollinearity.

3️⃣ Outliers

Outliers are extreme values that are very different from other data points.

🔎 Example:

If most house prices are:
100k – 300k

But one house is:
5 million

That 5 million value is an outlier.

Why it matters:
Outliers can:

Distort mean values
Mislead ML models
Increase error
They must be detected and handled carefully.
## 📌 Observations

- OverallQual and GrLivArea show strong positive correlation with SalePrice.
- SalePrice distribution is right-skewed.
- There are high-price outliers visible in the distribution.
- Visualization helps identify important predictive features.