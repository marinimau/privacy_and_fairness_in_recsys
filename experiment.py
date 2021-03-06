#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import conf
import os
import pandas as pd
import numpy as np

from load_data import get_gender_labels, get_age_labels
from classify import split_data, perform_classification
from evaluator import get_evaluation_metrics, get_confusion_matrix
from load_data import get_best_recs
from preprocessing import balance_data, merge_data, merge_embeddings, do_temporal_splitting
from utils import write_metrics, get_inspector_file_name, get_obfuscation_method
from conf import best


def run_experiment(original_data, label_name='gender', embeddings=False, recs=False, already_merged=False):
    """
    run an experiment
    :param original_data: the interaction, relevance or recs
    :param label_name: the label name
    :param embeddings: flag that requires special merge for embeddings
    :param recs: flag that requires special merge for recs
    :param already_merged: data is already merged
    :return:
    """
    if not already_merged:
        # get labels
        user_data = get_gender_labels() if label_name == 'gender' else get_age_labels()
        # balance data
        if conf.balance_data:
            user_data = balance_data(user_data, label_name)
        # merge labels with original data
        joined_df = merge_data(user_data, original_data, on_recs=recs) if not embeddings \
            else merge_embeddings(user_data, original_data)
    else:
        joined_df = original_data
    if conf.lite_dataset:
        joined_df = joined_df.head(conf.lite_dataset_size)
    # split in training and test set
    x_train, x_test, y_train, y_test = split_data(joined_df, preserve_order=conf.maintain_order_in_train_test_split)
    # classify
    y_random_forest, training_time_rf, prediction_time_rf = perform_classification(x_train, y_train, x_test,
                                                                                   'random_forest')
    y_logistic_regression, training_time_lr, prediction_time_lr = perform_classification(x_train, y_train, x_test,
                                                                                         'logistic_regression')
    # get evaluation metrics
    random_forest_metrics = get_evaluation_metrics(y_test, y_random_forest)
    logistic_regression_metrics = get_evaluation_metrics(y_test, y_logistic_regression)
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
    root_path = 'data_obfuscated' if conf.required_obfuscated['observation'] else 'data'
    file_path = root_path + '/' + conf.data_root + '/' + get_obfuscation_method(label) + 'ratings.tsv'
    if conf.VERBOSE:
        print("reading: " + file_path + " " + str(os.path.isfile(file_path)))
    df = pd.read_csv(file_path, header=None, sep='\t') if os.path.isfile(file_path) else None
    if df is not None:
        df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating', 3: 'timestamp'}, inplace=True)
        df.set_index('uid')
        if conf.data_root == conf.data_root_list[1]:
            df['rating'] = df['rating'].astype(int)
        # time cutoff
        if conf.perform_time_splitting:
            df = do_temporal_splitting(df)
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
    root_path = 'embeddings_obfuscated' if conf.required_obfuscated['embeddings'] else 'embeddings'
    file_path = './' + root_path + '/' + conf.data_root + '/' + get_obfuscation_method(label) + dataset_name + '.csv'
    if conf.VERBOSE:
        print("reading: " + file_path + " " + str(os.path.isfile(file_path)))
    df = pd.read_csv(file_path, header=None) if os.path.isfile(file_path) else None
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
    root_path = 'recs_obfuscated' if conf.required_obfuscated['recs'] else 'recs'
    file_path = './' + root_path + '/' + conf.data_root + '/' + get_obfuscation_method(label) + dataset_name + '.tsv'
    if conf.VERBOSE:
        print("reading: " + file_path + " " + str(os.path.isfile(file_path)))
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, header=None, sep='\t')
        df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating'}, inplace=True)
        df.set_index('uid')
    else:
        df = None
    # relevance
    relevance_dataset = dataset_name + '_relevance'
    conf.current_trade_off_file_name = get_inspector_file_name(relevance_dataset, label)
    metrics = get_metrics_from_classifier(df, label)
    write_metrics(label_name=label, dataset_name=relevance_dataset, metrics=metrics)
    # recs
    for i in best:
        recs = get_best_recs(df, i) if df is not None else None
        classification_dataset = dataset_name + '_classification_best_' + str(i)
        conf.current_trade_off_file_name = get_inspector_file_name(classification_dataset, label)
        metrics = get_metrics_from_classifier(recs, label)
        write_metrics(label_name=label, dataset_name=classification_dataset, metrics=metrics)


def get_metrics_from_classifier(df, label, embeddings=False, recs=False, already_merged=False):
    """
    get metrics from classifier
    :param df: the dataframe
    :param label: the label name
    :param embeddings: flag that requires special experiment for embeddings
    :param recs: flag that requires special merge for recs
    :param already_merged: data is already merged
    :return:
    """
    if conf.DEBUG or df is None:
        return [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0), (0, 0), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
    else:
        return run_experiment(df, label, embeddings=embeddings, recs=recs, already_merged=already_merged)
