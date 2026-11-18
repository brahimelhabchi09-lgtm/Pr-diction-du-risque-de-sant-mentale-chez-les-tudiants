# Student dataset

Download [Mental Health Risk Prediction Dataset (Students)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
and put `student_mental_health.csv` in `data/raw/`. The license is CC0 Public Domain.

From the project root, run:

```bash
python -m src.data
```

This reads the original file and saves an 80/20 stratified split with seed 42 in
`data/processed/X_train.csv`, `X_test.csv`, `y_train.csv` and `y_test.csv`.
Features and labels come directly from `train_test_split`. Raw and split CSVs are
ignored by Git. The raw file is not changed.

There are 2,000 simulated students and 17 columns. Drop Student_ID and predict
Mental_Health_Risk from the remaining 15 features. The classes are Low (660),
Medium (660) and High (680).

The EDA notebook checks missing values and unusual records. Use training medians
for numbers and training modes for categories. Keep valid extreme values. Fit
scaling and encoding inside the model pipelines. No resampling is needed.

Synthetic labels do not establish clinical validity.
