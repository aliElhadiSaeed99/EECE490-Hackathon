# Unknown Dataset Hackathon Submission

## Team Members
* Ali El Hadi Saeed
* Tamer Lammam
* Marc Bteish

## The approach to the hackathon:
Our program uses a universal machine learning pipeline built on `scikit-learn`. It reads an unfamiliar CSV, dynamically determines the task (Classification, Regression, or Clustering), and routes the data to the appropriate model (`RandomForestClassifier`, `RandomForestRegressor`, or `MiniBatchKMeans`). 

## Assumptions the model is based on:
1. **Irrelevant Features:** We assumed that any `object` column where the number of unique values equals the total number of rows is an identifier (e.g., customerID) and provides no predictive value. These are automatically dropped.
2. **Missing Values:** We assumed numerical missing values should be imputed using the median to avoid skew from outliers. Categorical missing values are filled with a constant "missing" label.
3. **Execution Limits:** To strictly follow the 60 second CPU limit, we utilized LightGBM algorithms and n_jobs=-1 to ensure maximum parallel processing efficiency.

## How to Reproduce
1. Install the pinned dependencies: `pip install -r requirements.txt`
2. Run the standard entry point with the target dataset: `python universal_pipeline.py path/to/dataset.csv`
