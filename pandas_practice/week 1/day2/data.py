import pandas as pd

df = pd.read_csv("data.csv")
print(df)

df.head()
df.info()
df.isnull().sum()

numeric_cols = df.select_dtypes(include="number").columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

    df.isnull().sum()

    houses_3_plus = df[df["Bedrooms"] > 3]
    houses_3_plus.head()
    houses_3_plus.shape