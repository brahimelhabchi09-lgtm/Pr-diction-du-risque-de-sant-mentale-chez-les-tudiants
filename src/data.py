import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import numeric_columns, categorical_columns


def prepare_data():
    df = pd.read_csv('data/raw/student_mental_health.csv')
    X = df[numeric_columns + categorical_columns]
    y = df['Mental_Health_Risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    print('Training rows:', len(X_train))
    print('Test rows:', len(X_test))


if __name__ == '__main__':
    prepare_data()
