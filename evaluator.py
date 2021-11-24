#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import conf


def get_evaluation_metrics(y_test, y_pred, binary=True):
    """
    Get accuracy metrics
    :param y_test: the real y labels
    :param y_pred: the predicted y labels
    :param binary: flag that indicates if the classification is binary
    :return: a tuple of metrics (balanced_accuracy, f1, precision, recall, roc_auc)
    """
    if binary:
        return balanced_accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test,
                                                                                                  y_pred), recall_score(
            y_test, y_pred), roc_auc_score(y_test, y_pred)
    else:
        return balanced_accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), \
               precision_score(y_test, y_pred, average='weighted', zero_division=1), \
               recall_score(y_test, y_pred, average='weighted'), \
               0 # roc_auc_score(y_test, y_pred, multi_class='ovo', average='macro', labels=conf.gender_labels)
