import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay,
)


def evaluate_model():
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Mental_Health_Risk']
    pipeline = joblib.load('models/best_pipeline.joblib')
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    scores = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision_macro': precision_score(y_test, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, predictions, average='weighted', zero_division=0),
        'roc_auc_ovr': roc_auc_score(y_test, probabilities, labels=pipeline.classes_, multi_class='ovr'),
        'recall_high': recall_score(y_test, predictions, labels=['High'], average='macro'),
    }
    pd.DataFrame([scores]).to_csv('models/test_metrics.csv', index=False)
    print(pd.Series(scores))
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv('models/classification_report.csv')
    print(classification_report(y_test, predictions, zero_division=0))
    print('Incorrect predictions:', (y_test != predictions).sum())
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=['Low', 'Medium', 'High'], cmap='Blues')
    plt.savefig('models/confusion_matrix.png')
    plt.close()


if __name__ == '__main__':
    evaluate_model()
