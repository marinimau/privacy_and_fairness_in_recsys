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
from load_data import get_best_recs
from utils import write_metrics
from conf import best


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
    random_forest_metrics = get_evaluation_metrics(y_test, y_random_forest, binary=(label_name == 'gender'))
    logistic_regression_metrics = get_evaluation_metrics(y_test, y_logistic_regression, binary=(label_name == 'gender'))
    return [random_forest_metrics, logistic_regression_metrics]


def experiment_automation(dataset, label):
    """
    auto run experiments
    :param dataset: the dataset name
    :param label: the label name
    :return:
    """
    # ratings
    df = pd.read_csv('./recs/' + dataset + '.tsv', header=None, sep='\t')
    df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating', 3: 'timestamp'}, inplace=True)
    df.set_index('uid')
    # relevance
    relevance_dataset = dataset + '_relevance'
    metrics = get_metrics_from_classifier(df, label)

    write_metrics(label_name=label, dataset_name=relevance_dataset, metrics=metrics)
    # recs
    for i in best:
        recs = get_best_recs(df, i)
        classification_dataset = dataset + '_classification_best_' + str(i)
        metrics = get_metrics_from_classifier(recs, label)
        write_metrics(label_name=label, dataset_name=classification_dataset, metrics=metrics)


def get_metrics_from_classifier(df, label):
    """
    get metrics from classifier
    :param df: the dataframe
    :param label: the label name
    :return:
    """
    if conf.DEBUG:
        return [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0)] if label == 'gender' else [[0], [0]]
    else:
        return run_experiment(df, label)
