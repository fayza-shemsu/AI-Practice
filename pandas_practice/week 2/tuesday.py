# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1️⃣ Load the dataset
# Make sure the CSV is in the correct folder (relative path)
df = pd.read_csv("data/customer_churn.csv")  # replace with your path if needed

# 2️⃣ Separate features and target
X = df.drop("Target", axis=1)  # all columns except target
y = df["Target"]               # Target: 1=Left, 0=Stayed

# 3️⃣ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 5️⃣ Make predictions
y_pred = log_model.predict(X_test)

# 6️⃣ Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 7️⃣ Print number of False Positives
FP = cm[0,1]
print(f"Number of False Positives: {FP}")

# 8️⃣ Optional: Print Training and Testing Accuracy
train_acc = log_model.score(X_train, y_train)
test_acc = log_model.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")