📘 Day 2 – Data Wrangling with Pandas
🎯 Objective
Understand DataFrames, Series, and data cleaning strategies using Pandas.

🐼 What Is Pandas?

Pandas is a Python library used for:

Data manipulation and data analysis.

It helps you:

*Read data (CSV, Excel, SQL)

*Clean data

*Transform data

*Analyze data

Prepare data for machine learning

🔹 Simple Definition

Pandas is a tool that allows you to work with structured data in Python using tables.

If NumPy works with numbers,
Pandas works with tables.

🔹 Real-World Example

Imagine you have an Excel sheet of house prices:

Price	Bedrooms	Location
200000	3	Addis
150000	2	Hawassa
300000	4	Bahir Dar

Pandas lets you:

*Load this file

*Clean missing values

*Filter houses

*Calculate averages

*Prepare it for ML

All inside Python.

🔹 Why It Is Called "Pandas"

The name comes from:Panel Data

Which is a term used in statistics for structured datasets.
It has nothing to do with the animal 😄

🔹 What Can Pandas Do?
1️⃣ Read Data
import pandas as pd
df = pd.read_csv("house_prices.csv")

Loads CSV into a table (DataFrame).

📘 1️⃣ Series vs DataFrame (Clearly Explained)

🔹 What is a Series?

A Series in Pandas is a one-dimensional labeled array.
That means:

## It holds a single column of data.

## Each value has an index (label).

Example:

import pandas as pd

s = pd.Series([100000, 150000, 200000])
print(s)
Output:

0    100000
1    150000
2    200000

💡 Use of a Series

A Series is used when:

🔹 You want to work with one feature (one column).

🔹 You want to calculate statistics like:

Mean
Median
Standard deviation
You want to apply transformations to a single variable.

Example in ML:
If you want to analyze only the Price column:

df["Price"]
That returns a Series.

So:A Series represents one variable in your dataset.

🔹 What is a DataFrame?

A DataFrame is a two-dimensional labeled data structure.
That means:

*It has rows and columns.

*Each column is actually a Series.

8It is like a full table.

Example:
df = pd.DataFrame({
    "Price": [100000, 150000, 200000],
    "Bedrooms": [2, 3, 4]
})
print(df)
Output:

   Price  Bedrooms
0 100000         2
1 150000         3
2 200000         4

💡 Use of a DataFrame

A DataFrame is used when:

You work with the full dataset.

You want to: Filter rows

             🔹 Select multiple columns

             🔹 Clean data

              🔹 Prepare data for machine learning

In ML:
The DataFrame represents your entire dataset.
## A DataFrame represents the full dataset.
## A Series represents one feature inside that dataset.

📘 2️⃣ Data Cleaning Strategies (Clearly Defined + Why We Use Them)

Data cleaning is the process of fixing or removing incorrect, incomplete, or irrelevant data.Why?

Because:
Machine learning models learn from data.
If the data is bad, the model becomes bad.

Now let’s define each clearly.

🔹 1. Handle Missing Values
What It Means
Sometimes data is incomplete.
Example:

Price	Bedrooms
200000	3	
150000	NaN

“NaN” means missing value.

Why It Is Important

Most ML algorithms cannot handle missing values.

If you don’t fix them:Training will fail Or the model becomes inaccurate

Common Solutions

🔹 Replace numeric missing values with mean

🔹 Replace categorical missing values with mode

Or drop the row entirely

Example:

df.fillna(df.mean(numeric_only=True), inplace=True)

Use:

🔹 Keeps dataset complete

🔹 Prevents training errors

      🔹 2. Remove Duplicates
What It Means
Sometimes the same row appears twice.

Example:

Price	Bedrooms
200000	3
200000	3
Why It Is Important

Duplicates:Bias the model
Make certain values appear more important than they 
Fix:

df.drop_duplicates(inplace=True)
Use:

Ensures fair and unbiased training

🔹 3. Fix Incorrect Data Types
What It Means:Sometimes numbers are stored as text.

Example:

"200000"  ← stored as string instead of integer

Why It Is Important

ML models need numerical input.

If types are wrong:Calculations fail
                   Statistics become incorrect
Fix:

df["Price"] = df["Price"].astype(int)
Use:

Ensures correct mathematical operations

🔹 4. Filter Relevant Records
What It Means

Selecting only the data that matters.

Example:

df[df["Bedrooms"] > 3]

Why It Is Important

In ML:

You may want to focus on a specific segment.

You remove irrelevant data.

You reduce noise.

Use:

Improves model performance

Makes analysis clearer

🎯 Why Data Cleaning Is Critical in AI

Before building a model:Clean data
                    🔹 Analyze data
                    🔹 Engineer feature
                    🔹 Train model

If you skip cleaning:

      🔹 Model accuracy drops

      🔹Predictions become unreliable

Deployment fails

In real companies:
Data cleaning often takes 70–80% of project time