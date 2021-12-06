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
import numpy as np

from load_data import get_gender_labels, get_age_labels
from classify import split_data, perform_classification
from evaluator import get_evaluation_metrics, get_confusion_matrix
from load_data import get_best_recs
from preprocessing import balance_data, merge_data, merge_embeddings
from utils import write_metrics, get_inspector_file_name
from conf import best


def run_experiment(original_data, label_name='gender', embeddings=False):
    """
    run an experiment
    :param original_data: the interaction, relevance or recs
    :param label_name: the label name
    :param embeddings: flag that requires special experiment for embeddings
    :return:
    """
    # get labels
    user_data = get_gender_labels() if label_name == 'gender' else get_age_labels()
    # balance data
    if conf.balance_data and label_name == 'gender':
        user_data = balance_data(user_data)
    # merge labels with original data
    joined_df = merge_data(user_data, original_data) if not embeddings else merge_embeddings(user_data, original_data)
    if conf.lite_dataset:
        joined_df = joined_df.head(conf.lite_dataset_size)
    # split in training and test set
    x_train, x_test, y_train, y_test = split_data(joined_df)
    # classify
    y_random_forest, training_time_rf, prediction_time_rf = perform_classification(x_train, y_train, x_test,
                                                                                   'random_forest')
    y_logistic_regression, training_time_lr, prediction_time_lr = perform_classification(x_train, y_train, x_test,
                                                                                         'logistic_regression')
    # get evaluation metrics
    random_forest_metrics = get_evaluation_metrics(y_test, y_random_forest, binary=(label_name == 'gender'))
    logistic_regression_metrics = get_evaluation_metrics(y_test, y_logistic_regression, binary=(label_name == 'gender'))
    return [random_forest_metrics, logistic_regression_metrics, (training_time_rf, prediction_time_rf),
            (training_time_lr, prediction_time_lr), get_confusion_matrix(y_test, y_random_forest),
            get_confusion_matrix(y_test, y_logistic_regression)]


def observation_experiment(label):
    """
    auto run observation experiments
    :param label: the label name
    :return:
    """
    # interactions
    df = pd.read_csv('./data/ratings.tsv', header=None, sep='\t')
    df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating', 3: 'timestamp'}, inplace=True)
    df.set_index('uid')
    metrics = get_metrics_from_classifier(df, label)
    write_metrics(label_name=label, dataset_name='observation', metrics=metrics)


def embedding_experiments(dataset_name, label):
    """
    auto run experiments for embeddings step
    :param dataset_name: the dataset name
    :param label: the label name
    :return:
    """
    # ratings
    if dataset_name == 'most-pop':
        df = None
    else:
        df = pd.read_csv('./embeddings/' + dataset_name + '.csv', header=None)
    embeddings_dataset = dataset_name + '_embeddings'
    conf.current_trade_off_file_name = get_inspector_file_name(embeddings_dataset, label)
    metrics = get_metrics_from_classifier(df, label, embeddings=True)
    write_metrics(label_name=label, dataset_name=embeddings_dataset, metrics=metrics)


def recs_experiment(dataset_name, label):
    """
    auto run experiments for relevance and classification step
    :param dataset_name: the dataset name
    :param label: the label name
    :return:
    """
    # ratings
    df = pd.read_csv('./recs/' + dataset_name + '.tsv', header=None, sep='\t')
    df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating'}, inplace=True)
    df.set_index('uid')
    # relevance
    relevance_dataset = dataset_name + '_relevance'
    conf.current_trade_off_file_name = get_inspector_file_name(relevance_dataset, label)
    metrics = get_metrics_from_classifier(df, label)
    write_metrics(label_name=label, dataset_name=relevance_dataset, metrics=metrics)
    # recs
    for i in best:
        recs = get_best_recs(df, i)
        classification_dataset = dataset_name + '_classification_best_' + str(i)
        conf.current_trade_off_file_name = get_inspector_file_name(classification_dataset, label)
        metrics = get_metrics_from_classifier(recs, label)
        write_metrics(label_name=label, dataset_name=classification_dataset, metrics=metrics)


def get_metrics_from_classifier(df, label, embeddings=False):
    """
    get metrics from classifier
    :param df: the dataframe
    :param label: the label name
    :param embeddings: flag that requires special experiment for embeddings
    :return:
    """
    if conf.DEBUG or df is None:
        return [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0), (0, 0), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
    else:
        return run_experiment(df, label, embeddings)
