PIPELINE DIAGRAM
================

Input: churn-dataset (azureml:churn-dataset@latest)
       type: URI_FILE — a single CSV file

       │
       ▼

┌──────────────────────────────────────┐
│  STEP 1: data_prep                   │
│  Script: data_prep.py                │
│  Input:  raw_data (URI_FILE)         │
│  Output: cleaned_data (URI_FOLDER)   │
│  Does:   drops CustomerID,           │
│          removes duplicates          │
└──────────────┬───────────────────────┘
               │ cleaned.csv
               │
       ┌───────┴───────┐
       │               │
       ▼               │
┌──────────────────┐   │ reused directly
│  STEP 2: train   │   │
│  Script: train.py│   │
│  Input:  cleaned │   │
│  Output: model   │   │
│  Does:   encodes,│   │
│          fits RF │   │
└──────┬───────────┘   │
       │ model.pkl      │
       ▼               ▼
┌──────────────────────────────────────┐
│  STEP 3: evaluate                    │
│  Script: evaluate.py                 │
│  Input:  model (URI_FOLDER)          │
│          cleaned_data (URI_FOLDER)   │
│  Output: F1 score printed to logs    │
│  Does:   reindex columns, predict    │
└──────────────────────────────────────┘

WHY THIS IS A DAG
- Directed: arrows go one way only
- Acyclic: no step depends on itself
- Graph: evaluate has two parents

REPRODUCIBILITY
- Every run gets unique ID
- same code + same data + same env = identical result
- random_state=42 makes RandomForest deterministic