# Model results and limitations

All candidates used the same 1,600 training rows, fold-fitted preprocessing and
five stratified folds, seed 42. Selection used mean **macro F1** only.

| Model | Baseline CV macro F1 | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.7473 | 0.0213 |
| Gradient Boosting (histogram) | 0.6787 | 0.0239 |
| Random Forest | 0.6670 | 0.0224 |

GridSearchCV compared six Logistic Regression combinations (C=0.1/1/10,
class_weight=None/balanced) and eight Gradient Boosting combinations
(learning_rate=0.05/0.1, max_leaf_nodes=7/15, l2_regularization=0/1).
Tuned CV macro F1: **0.7502** and **0.6950**, respectively. Selected Logistic
Regression uses C=1, class_weight=balanced, max_iter=2000. Class weighting was
searched despite near balance; its small improvement may reflect fold noise.
The linear model's performance is consistent with the documented weighted
synthetic label rule; it does not establish how labels were actually generated.

The independent 400-row holdout was evaluated after selection, with no retuning:

| Metric | Holdout |
|---|---:|
| Accuracy | 0.7550 |
| Macro precision | 0.7540 |
| Macro recall | 0.7543 |
| Macro F1 | 0.7541 |
| Weighted F1 | 0.7548 |
| Macro one-vs-rest ROC-AUC | 0.9139 |
| High-label recall | 0.8235 |

Confusion matrix (rows=true, columns=predicted, Low/Medium/High):

| True label | Low | Medium | High |
|---|---:|---:|---:|
| Low | 107 | 24 | 1 |
| Medium | 24 | 83 | 25 |
| High | 0 | 24 | 112 |

Of 98 mistakes, 97 are adjacent-label confusions. Medium recall is only 83/132
(62.88%): this synthetic middle category overlaps both neighbors. Twenty-four
High records were predicted Medium; 26 other records were predicted High.
Noise in the generator and missing survey values plausibly contribute, but the
available data does not establish the exact cause of individual errors.

Training-only permutation importance (five repeats, macro F1 decrease) ranks
sleep (0.2502), financial stress (0.1356), family support (0.0960), previous
history (0.0913) and activity (0.0705) highest. These are descriptive in-sample
associations. Correlated features can share importance; these are not causal
estimates or independent validation results.

Gender-group holdout accuracy is 79.17% (Female, n=192), 70.97% (Male, n=186),
81.82% (Non-binary, n=22). The group gap merits attention, but especially the
22-record group is too small to establish fairness. No intervention or fairness
claim is justified. Full per-class and group metrics are in `models/metrics.json`.

The data is synthetic and labels reportedly derive from a weighted rule over
predictors plus noise. The exact generator is unavailable. These scores show
recovery of synthetic labels only, with no real-world clinical validation.
Non-nested model selection and one fixed holdout limit certainty. Class scores
are uncalibrated and are not medical probabilities. Repeated scheduled runs use
the same split and do not create new independent validation evidence.

The saved `best_pipeline.joblib` includes imputation, encoding, scaling and the
classifier. `scaler.joblib` exports the same fitted numeric scaler for the brief;
serving always uses the complete pipeline. No additional input scaling is needed.
Artifacts were loaded and used successfully in a fresh Python 3.12 process.
