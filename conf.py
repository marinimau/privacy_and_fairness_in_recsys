#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 08/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#


urls = {
    "movielens1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
}

classifiers = ['random-forest', 'logistic-regression']

dataset_names = [
    'observation',
    'observation-no-score',
    'most-pop_relevance',
    'most-pop_classification',
    'item-knn_relevance',
    'item-knn_classification',
    'user-knn_relevance',
    'user-knn_classification',
    'bprmf_relevance',
    'bprmf-knn_classification',
]

label_names = ['age', 'gender']

metrics = ['balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score']

VERBOSE = True

