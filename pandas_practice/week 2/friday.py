from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200, 300]
}
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,              # 5-fold cross validation
    scoring="accuracy",
    n_jobs=-1          # use all CPU cores
)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Tuned Random Forest Accuracy: {accuracy:.2%}")
