import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
data = {
    'Color': ['Red', 'Blue', 'Green', 'Red'],
    'Salary': [50000, 60000, 55000, 65000],
    'Age': [25, 30, 28, 35]
}

df = pd.DataFrame(data)
print(df)
def preprocess_data(df, categorical_cols, numeric_cols):
    """
    Preprocess a DataFrame:
    - OneHotEncode categorical columns
    - Standard scale numeric columns
    Returns a processed DataFrame.
    """
    # 1️⃣ One-Hot Encode categorical columns
    ohe = OneHotEncoder(sparse_output=False, drop=None)
    cat_encoded = ohe.fit_transform(df[categorical_cols])
    cat_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(categorical_cols))
    
    # 2️⃣ Scale numeric columns
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[numeric_cols])
    num_df = pd.DataFrame(num_scaled, columns=numeric_cols)
    
    # 3️⃣ Combine numeric and categorical columns
    processed_df = pd.concat([num_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    
    return processed_df
categorical_cols = ['Color']
numeric_cols = ['Salary', 'Age']

processed_df = preprocess_data(df, categorical_cols, numeric_cols)
print(processed_df)