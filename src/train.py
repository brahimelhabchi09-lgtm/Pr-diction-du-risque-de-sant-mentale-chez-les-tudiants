import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocessor


def train_models():
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['Mental_Health_Risk']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, early_stopping=False, random_state=42),
    }
    results = []
    for name, model in models.items():
        pipeline = Pipeline([('preprocess', preprocessor), ('model', model)])
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=['accuracy', 'f1_macro']
        )
        results.append({
            'model': name,
            'cv_accuracy': scores['test_accuracy'].mean(),
            'cv_f1': scores['test_f1_macro'].mean(),
            'cv_std': scores['test_f1_macro'].std(),
        })
    comparison = pd.DataFrame(results).sort_values('cv_f1', ascending=False)
    comparison.to_csv('models/comparison.csv', index=False)
    print(comparison)

    grids = {
        'Logistic Regression': {'model__C': [0.1, 1., 10.], 'model__class_weight': [None, 'balanced']},
        'Random Forest': {'model__max_depth': [None, 8, 16], 'model__min_samples_leaf': [1, 3]},
        'Gradient Boosting': {'model__learning_rate': [0.05, 0.1],
                              'model__max_leaf_nodes': [7, 15], 'model__l2_regularization': [0., 1.]},
    }
    results = []
    pipelines = {}
    for name in comparison.head(2)['model']:
        pipeline = Pipeline([('preprocess', preprocessor), ('model', models[name])])
        search = GridSearchCV(pipeline, grids[name], scoring='f1_macro', cv=cv)
        search.fit(X_train, y_train)
        pipelines[name] = search.best_estimator_
        results.append({
            'model': name, 'cv_f1': search.best_score_,
            'parameters': str(search.best_params_),
        })
    tuning = pd.DataFrame(results).sort_values('cv_f1', ascending=False)
    tuning.to_csv('models/tuning.csv', index=False)
    print(tuning)

    best_model = tuning.iloc[0]['model']
    best_pipeline = pipelines[best_model]
    joblib.dump(best_pipeline, 'models/best_pipeline.joblib', compress=3)
    numeric_pipeline = best_pipeline.named_steps['preprocess'].named_transformers_['numeric']
    joblib.dump(numeric_pipeline.named_steps['scale'], 'models/scaler.joblib', compress=3)
    print('Selected model:', best_model)


if __name__ == '__main__':
    train_models()
