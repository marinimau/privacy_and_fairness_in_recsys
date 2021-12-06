#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 08/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

urls = {
    "movielens1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
}

classifiers = ['random-forest', 'logistic-regression']

classifier_models = {
    "random_forest": RandomForestClassifier(),
    "logistic_regression": LogisticRegression(),

}

classifier_params = {
    'random_forest': {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'n_jobs': [-1],
        'min_samples_leaf': [4, 40, 100, 200],
        'random_state': [101, 698]
    },
    'logistic_regression': {
        'penalty': ['l2'],
        'C': [1.0],
        'random_state': [np.random.RandomState(0)],
    }
}

trade_off_param_range = {
    'random_forest': np.arange(200, 700, 100),
    'logistic_regression': [0.001, 0.05, 0.1, 0.5, 1.0, 10.0]
}

trade_off_param_name = {
    'random_forest': 'n_estimators',
    'logistic_regression': 'C'
}

best = [10, 20, 50]

recs_file_names = [
    'user-knn',
    'item-knn',
    'most-pop',
    'bprmf',
]

dataset_names = [
    'observation',
    'observation-no-score',
    'most-pop_relevance',
    'most-pop_embeddings',
    'most-pop_classification_best_10',
    'most-pop_classification_best_20',
    'most-pop_classification_best_50',
    'item-knn_relevance',
    'item-knn_embeddings',
    'item-knn_classification_best_10',
    'item-knn_classification_best_20',
    'item-knn_classification_best_50',
    'user-knn_relevance',
    'user-knn_embeddings',
    'user-knn_classification_best_10',
    'user-knn_classification_best_20',
    'user-knn_classification_best_50',
    'bprmf_relevance',
    'bprmf_embeddings',
    'bprmf_classification_best_10',
    'bprmf_classification_best_20',
    'bprmf_classification_best_50',
]

label_names = ['gender', 'age']

gender_labels = ['1', '56', '25', '45', '50', '35', '18']

metrics = ['balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'accuracy_score']

metrics_not_binary = ['accuracy']

balance_data = True

lite_dataset = False

lite_dataset_size = 100

classifier_evaluation_plot = True

VERBOSE = True

SHOW_PLOT = False

DEBUG = False

current_trade_off_file_name = ''
