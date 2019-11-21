
"""
Created on Mon Oct  7
@author: Miriam
"""

print('hi')

## Setups

import os

import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm

# import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier # For Classification
from tabulate import tabulate
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance

dataset = pd.read_csv('dataset.csv')
dataset = pd.read_csv('dataset_full.csv')

## Data checking/visualisation

def print_full(dataframe):
    """Print full columns and rows."""
    with pd.option_context('display.max_columns', None,
                           'display.max_rows', None):
        print(dataframe)

dataset.shape        
print_full(dataset.head(3))
dataset.columns = ['age', 'job', 'marital', 'education', 'default',
                   'balance', 'housing', 'loan', 'contact',
                   'day', 'month', 'duration', 'campaign',
                   'pdays', 'previous', 'poutcome',
                   'deposit']

dataset.corr().deposit
dataset.apply(set)
dataset.apply(lambda x: sum(x == 'unknown'))

sum(dataset.deposit==1)
sum(dataset.deposit==2) # highly imbalanced

# Visualization

sns.pairplot(dataset)

## Feature Engineering

dataset.corr().deposit
print_full(dataset.corr())

del dataset["duration"]

# One-hot Encoding

dataset_dummies = pd.get_dummies(dataset)
dataset2 = pd.get_dummies(dataset)
print_full(dataset_dummies.head(5))
dataset_dummies.shape

correlations = abs(dataset_dummies.corr())
correlations[correlations > 0.05].deposit.sort_values().dropna()

correlations = abs(dataset_dummies.cov())
correlations.deposit.sort_values().dropna()

## Modelling

# splitting 

y = dataset2.deposit
X = dataset2.drop(columns="deposit")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,
                                                    random_state = 0)

# Gradient Boosting (GBM)

clf = GradientBoostingClassifier()

clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_test)
classification_report(y_pred2, y_test)
print('              precision    recall  f1-score   support\n\n           1       0.99      0.90      0.94       441\n           2       0.16      0.67      0.25        12\n\n    accuracy                           0.90       453\n   macro avg       0.57      0.78      0.60       453\nweighted avg       0.97      0.90      0.93       453\n')

clf.feature_importances_
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)

# Extreme Gradient Boosting (XGBM)

model = XGBClassifier()
model.fit(X_train, y_train)
plot_importance(model)


y_pred = model.predict(X_test)
classification_report(y_pred, y_test)
print('              precision    recall  f1-score   support\n\n           1       0.99      0.90      0.94       443\n           2       0.14      0.70      0.23        10\n\n    accuracy                           0.90       453\n   macro avg       0.56      0.80      0.59       453\nweighted avg       0.97      0.90      0.93       453\n')

# Light Gradient Boosting (LGBM)

categorical_features = [c for c, col in enumerate(dataset.columns) if 'cat' in col]

train_data = lightgbm.Dataset(X_train, label=y_train,
                              categorical_feature=categorical_features)
test_data = lightgbm.Dataset(X_test, label=y_test)

parameters = {
        'application': 'binary',
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 2,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 0
        }

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=20,
                       early_stopping_rounds=100)

y_pred = model.predict(X_test)
y_pred

unique, counts = np.unique(y_pred, return_counts=True)
dict(zip(unique, counts))

## Validation
