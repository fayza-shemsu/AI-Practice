# 🚀 AI Engineer Intern Journey — Week 1 & 2

## 📌 Overview

This repository documents my journey from scratch to building a **production-style Machine Learning pipeline** for a **Customer Churn Prediction System**.

The goal was not just to train models, but to:

* Understand the **full ML lifecycle**
* Build **modular, reusable pipelines**
* Apply **feature engineering, model selection, and tuning**
* Follow **real-world engineering practices**

---

# 🧠 What I Built (Big Picture)

I developed a complete ML system:

```
Raw Data → Cleaning → Feature Engineering → Model Training → Evaluation → Model Selection → Hyperparameter Tuning → Final Model
```

---

# 📂 Project Structure (Explained Like a Senior Engineer)

```
faiz_ai_journey/
│
├── data/
│   ├── raw/              # Original datasets (never modified)
│   ├── processed/        # Cleaned & transformed data
│
├── models/
│   ├── saved_models/     # Intermediate models
│   ├── best_model.pkl    # Best selected model
│   ├── churn_pipeline.pkl
│   ├── tuned_random_forest.pkl
│
├── notebooks/
│   ├── week1/            # Foundations (Pandas, visualization)
│   ├── week2/            # ML models & tuning
│
├── src/
│   ├── data/             # Data loading logic
│   ├── features/         # Feature engineering
│   │   ├── feature_engineering.py
│   │   ├── transformers.py
│   │
│   ├── models/
│   │   ├── train_pipeline.py
│   │   ├── classification_pipeline.py
│   │   ├── model_selection.py
│   │   ├── hyperparameter_tuning.py
│   │
│   ├── pipelines/        # End-to-end ML pipelines
│   ├── utils/            # Helper functions
│
├── reports/              # Metrics & outputs
├── tests/                # Testing (future use)
│
├── requirements.txt
├── README.md
```

---

# 🔵 WEEK 1 — FOUNDATION

## 1. Data Handling (Pandas)

### What I did:

* Loaded CSV datasets
* Handled missing values:

  * Mean (numerical)
  * Mode (categorical)
* Filtered and explored data

### Why it matters:

Real-world data is:

* Messy
* Incomplete
* Inconsistent

👉 Cleaning = **80% of ML work**

---

## 2. Data Visualization

### Tools used:

* Matplotlib
* Seaborn

### What I analyzed:

* Distribution of features
* Correlation heatmaps
* Outliers using boxplots

### Why:

👉 Understand patterns BEFORE modeling

---

## 3. First Model (Linear & Basic ML)

### Concepts learned:

* Features (X) vs Target (y)
* Train/Test split
* Model training & prediction

---

## 4. Evaluation Metrics

### Used:

* MAE, RMSE, R² (Regression)
* Accuracy (initial understanding)

### Key Insight:

👉 Metrics define **how we judge models**

---

# 🔵 WEEK 2 — REAL MACHINE LEARNING

---

## 1. Feature Engineering

### What I implemented:

* OneHotEncoding (categorical → numeric)
* StandardScaler (feature scaling)

### Why:

👉 Models only understand numbers
👉 Scaling prevents feature dominance

---

## 2. Classification System (Churn Prediction)

### Model:

* Logistic Regression

### Key Concepts:

* Sigmoid function → probability output
* Threshold → decision boundary

### Real-world thinking:

👉 Predict **risk**, not just labels

---

## 3. Tree-Based Models

### Models:

* Decision Tree
* Random Forest

### Why Random Forest:

* Handles non-linear data
* Reduces overfitting via ensembling

---

## 4. Model Comparison Framework

### Built system to compare:

* Logistic Regression
* Random Forest
* SVM

### Key Features:

* Unified evaluation pipeline
* Performance tracking
* Leaderboard generation

### Insight:

👉 No single model is always best (**No Free Lunch Theorem**)

---

## 5. Hyperparameter Tuning

### Implemented:

* GridSearchCV

### Tuned:

* n_estimators
* max_depth
* min_samples_split

### Result:

* Significant performance improvement
* Final optimized model saved

---

## 6. Final Output

### Saved models:

* `tuned_random_forest.pkl`
* `best_model.pkl`

### Evaluation:

* Confusion Matrix
* Precision / Recall / F1 Score

---

# 🔥 KEY ENGINEERING PRINCIPLES LEARNED

---

## 1. Pipeline Thinking

Instead of:

```
Random scripts
```

I built:

```
Reusable, structured pipelines
```

---

## 2. Separation of Concerns

| Layer     | Responsibility     |
| --------- | ------------------ |
| Data      | Loading & cleaning |
| Features  | Transformation     |
| Models    | Training           |
| Pipelines | Orchestration      |

---

## 3. Reproducibility

* Same code → same results
* Controlled workflows

---

## 4. Model Evaluation Mindset

Not just:

```
Accuracy
```

But:

* Precision
* Recall
* F1 Score
* Overfitting detection

---

## 5. Real-World Thinking

Every model decision tied to:

* Business impact
* Trade-offs (FP vs FN)

---

# 📊 FINAL SYSTEM (END-TO-END)

```
Raw CSV
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Pipeline
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Best Model Saved (.pkl)
```

---

# 🧭 WHAT THIS PREPARES ME FOR

After Week 2, I can:

✅ Build full ML pipelines
✅ Compare multiple models
✅ Tune performance
✅ Structure projects professionally
✅ Think like an AI engineer (not just coder)

---

# 🚀 NEXT STEP (WEEK 3)

Move from:

```
Local ML
```

to:

```
Cloud ML (Azure ML)
```

Where I will:

* Manage data as assets
* Run models on cloud compute
* Build scalable ML systems

---

# 💡 FINAL NOTE

This project is not just about models.

It is about:

> 🔥 Building reliable, scalable, and production-ready AI systems
