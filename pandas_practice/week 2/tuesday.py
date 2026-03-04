# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset  
df = pd.read_csv("customer_churn.csv")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Build pipeline with Logistic Regression
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(solver='liblinear'))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])  # [1,0] so rows: Left, Stayed
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Left", "Stayed"])
disp.plot()

# Inspect False Positives
# False Positives: Predicted Left (1) but actually Stayed (0)
FP = cm[0, 1]
print(f"False Positives (Predicted Left but Stayed): {FP}")