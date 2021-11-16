#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import conf
import pandas as pd

from load_data import get_gender_labels, get_age_labels
from classify import split_data, perform_random_forest, perform_logistic_regression
from evaluator import get_evaluation_metrics


def run_experiment(original_data, label_name='gender'):
    """
    run an experiment
    :param original_data: the interaction, relevance or recs
    :param label_name: the label name
    :return:
    """
    assert label_name in conf.label_names
    # get labels
    user_data = get_gender_labels() if label_name == 'gender' else get_age_labels()
    # merge labels with original data
    joined_df = pd.merge(user_data, original_data, on='uid')
    # split in training and test set
    x_train, x_test, y_train, y_test = split_data(joined_df)
    # classify
    y_random_forest = perform_random_forest(x_train, y_train, x_test)
    y_logistic_regression = perform_logistic_regression(x_train, y_train, x_test)
    # get evaluation metrics
    random_forest_metrics = get_evaluation_metrics(y_test, y_random_forest)
    logistic_regression_metrics = get_evaluation_metrics(y_test, y_logistic_regression)
    return random_forest_metrics, logistic_regression_metrics