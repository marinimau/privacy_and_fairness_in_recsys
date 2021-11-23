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
    'most-pop_classification_best_10',
    'most-pop_classification_best_20',
    'most-pop_classification_best_50',
    'item-knn_relevance',
    'item-knn_classification_best_10',
    'item-knn_classification_best_20',
    'item-knn_classification_best_50',
    'user-knn_relevance',
    'user-knn_classification_best_10',
    'user-knn_classification_best_20',
    'user-knn_classification_best_50',
    'bprmf_relevance',
    'bprmf_classification_best_10',
    'bprmf_classification_best_20',
    'bprmf_classification_best_50',
]

label_names = ['gender', 'age']

metrics = ['balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score']

metrics_not_binary = ['accuracy']

balance_data = True

lite_dataset = True

lite_dataset_size = 10000

classifier_evaluation_plot = False

VERBOSE = True

SHOW_PLOT = False

DEBUG = False



