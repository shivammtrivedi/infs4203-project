"""INFS4203/7203 project - final pipeline (credit-card default classifier).

Run from the submission root, with development.csv and test.csv in the same folder:
    python main.py

It (1) reports the 5-fold stratified cross-validation metrics of the final pipeline on development.csv,
(2) refits the same pipeline on all of development.csv, (3) predicts test.csv in its original row order and
(4) writes s4802160.infs4203 (6,000 'yes,'/'no,' rows + one metrics row).
All exploration, comparison and tuning that led to these choices is in development.ipynb: the forest's
hyperparameters come from the GridSearchCV in Sections 6 and 6.1, and the per-client imputation from Section 6.2.
"""
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
STUDENT_ID = 's4802160'
HERE = os.path.dirname(os.path.abspath(__file__))

MONTHS = ['oct', 'nov', 'dec', 'jan', 'feb', 'mar']
TARGET = 'defaulted_next_month'
BILL_COLS = [f'bill_amount_{m}' for m in MONTHS]
PAYMENT_COLS = [f'payment_amount_{m}' for m in MONTHS]
NUMERIC_COLS = ['credit_limit', 'age'] + BILL_COLS + PAYMENT_COLS
CATEGORICAL_COLS = ['sex', 'education', 'marital_status']
REPAYMENT_COLS = [f'repayment_status_{m}' for m in MONTHS]
PREDICTOR_COLS = NUMERIC_COLS + CATEGORICAL_COLS + REPAYMENT_COLS     # client_id is never used

# Repayment status as an ordinal number: the three "not delayed" statuses share level 0, delays count months.
# Fixed hand-written mapping (nothing is learned from the data), so no information leaks between folds.
ORDINAL_MAP = {'no_balance': 0, 'paid_on_time': 0, 'revolving_credit': 0}
ORDINAL_MAP.update({f'payment_delayed_{k}_month' + ('' if k == 1 else 's'): k for k in range(1, 9)})


class PerClientImputer(BaseEstimator, TransformerMixin):
    """Per-client imputation by an aggregate value (Lecture 2, but taken over the client's OWN other months).
    A missing bill (or payment) is replaced by the mean of that same client's observed bills (or payments) in
    the other months. If the client has none observed, the training-fold column mean is used. No labels and no
    other clients are involved when transforming, so it applies identically to unseen rows."""

    def __init__(self, groups=()):
        self.groups = groups            # tuple of column-name tuples, each imputed from the row's own values

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.fallback_ = X.mean()       # fitted on the training fold only
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for cols in self.groups:
            row_mean = X[list(cols)].mean(axis=1)
            for col in cols:
                X[col] = X[col].fillna(row_mean)
        return X.fillna(self.fallback_).values


def load_features(path):
    frame = pd.read_csv(path)
    features = frame[PREDICTOR_COLS].copy()
    for col in REPAYMENT_COLS:
        features[col] = features[col].map(ORDINAL_MAP)
    return frame, features


def build_pipeline():
    """Per-client imputation for bills and payments (other numerics: column mean), mean imputation for the
    ordinal repayment columns, most-frequent imputation + one-hot for the categorical columns, then a
    class-weighted random forest. Refit inside every training fold by cross_validate."""
    preprocess = ColumnTransformer([
        ('num', Pipeline([('impute', PerClientImputer(groups=(tuple(BILL_COLS), tuple(PAYMENT_COLS))))]), NUMERIC_COLS),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_COLS),
        ('rep', Pipeline([('impute', SimpleImputer(strategy='mean'))]), REPAYMENT_COLS),
    ])
    forest = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=10, criterion='gini',
                                    class_weight='balanced', random_state=SEED)
    return Pipeline([('preprocess', preprocess), ('clf', forest)])


def main():
    dev_raw, X_dev = load_features(os.path.join(HERE, 'development.csv'))
    test_raw, X_test = load_features(os.path.join(HERE, 'test.csv'))
    y_dev = (dev_raw[TARGET] == 'yes').astype(int)          # 1 = default (positive class)

    # 1) Validation metrics: stratified 5-fold CV, held-out folds only, preprocessing refit per training fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro']
    scores = cross_validate(build_pipeline(), X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=-1)
    means = [scores['test_' + m].mean() for m in scoring]
    stds = [scores['test_' + m].std() for m in scoring]
    for name, mean, std in zip(['accuracy', 'precision (yes)', 'recall (yes)', 'F1 (yes)', 'macro-F1'], means, stds):
        print(f'{name:16s} {mean:.4f} +- {std:.4f}')

    # 2) Refit on all of development.csv, 3) predict test.csv in original row order
    final_model = build_pipeline().fit(X_dev, y_dev)
    predictions = final_model.predict(X_test)

    # 4) Write the submission file: 6,000 prediction rows + 1 metrics row, no header
    lines = [('yes,' if p == 1 else 'no,') for p in predictions]
    lines.append(','.join(f'{m:.3f}' for m in means) + ',')
    out_path = os.path.join(HERE, f'{STUDENT_ID}.infs4203')
    with open(out_path, 'w', newline='') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {out_path} ({len(lines)} rows; {int(predictions.sum())} of {len(predictions)} predicted yes)')


if __name__ == '__main__':
    main()
