## Tuesday: Data Wrangling with Pandas
- Loaded train.csv
- Checked for missing values and filled them (mean for numeric, mode for categorical)
- Filtered houses with more than 3 bedrooms
- Created `pandas_tasks.py`

## Wednesday: Data Storytelling (Visualization)
- Used Seaborn and Matplotlib to analyze `SalePrice`
- Plotted correlation heatmap to see which features are related to price
- Plotted distribution histogram of SalePrice
- Identified outliers using boxplots
- Created `visualization.py`

## Thursday: Your First Model (Scikit-Learn)

**Theory:** Supervised Learning, Features (X) vs Target (y), Train/Test Split.

**Practice:**
- Split data: `train_test_split(X, y, test_size=0.2)`
- Imported `LinearRegression` from `sklearn.linear_model`
- Trained the model: `model.fit(X_train, y_train)`
- Predicted: `preds = model.predict(X_test)`
- Evaluated with Mean Squared Error: `1359611455.66`
