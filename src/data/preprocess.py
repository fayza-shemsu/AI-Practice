import pandas as pd

def clean_data(df):
    import numpy as np

    # Simulate missing (for learning)
    df.loc[0, 'area'] = np.nan
    df.loc[1, 'furnishingstatus'] = np.nan

    # Separate columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Fill missing
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode yes/no
    cols = [
        'mainroad','guestroom','basement',
        'hotwaterheating','airconditioning','prefarea'
    ]

    for col in cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    return df