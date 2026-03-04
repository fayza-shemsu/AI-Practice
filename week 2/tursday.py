from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVM": SVC()
}
scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    scores[name] = accuracy
    for model, score in scores.items():

     print(f"{model}: {score:.2%} accuracy")
     best_model = max(scores, key=scores.get)
     best_score = scores[best_model]

    print(f"\n🏆 Best Model is {best_model} with {best_score:.2%} accuracy")