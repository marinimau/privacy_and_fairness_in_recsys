#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import time
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

import conf
from inspector import ClassifierTradeOffInspector


def split_data(dff, test_size=0.3, random_state=698, preserve_order=False):
    """
    Split data for classification phase
    :param dff: the original dataframe
    :param test_size: the percentage for test set splitting
    :param random_state: the random state
    :param preserve_order: preserve the order of the user
    :return:
    """
    x = dff.iloc[:, dff.columns != 'class']
    y = dff.iloc[:, dff.columns == 'class']
    if conf.normalize_data:
        x = pd.DataFrame(preprocessing.normalize(x)).fillna(0)
    if not preserve_order:
        return train_test_split(x, y, test_size=test_size, random_state=random_state)
    else:
        x_train, y_train = x[0:int(0.8 * len(x))], y[0:int(0.8 * len(x))]
        x_test, y_test = x[int(0.8 * len(x)):], y[int(0.8 * len(x)):]
        return x_train, x_test, y_train, y_test


def perform_classification(x_train, y_train, x_test, classifier_name, pred_proba=False):
    """
    Make predictions with the given classifier
    :param x_train: the x of training set
    :param y_train: the y of the training set (label)
    :param x_test: the x of the test set
    :param classifier_name: the classifier
    :param pred_proba: use predict_proba instead predict
    :return:
    """
    param_grid = conf.classifier_params[classifier_name]
    clf = conf.classifier_models[classifier_name]
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
    start = time.time()
    grid_search.fit(x_train.values, y_train.values.ravel())
    end = time.time()
    training_time = end - start
    if conf.VERBOSE:
        print(grid_search.best_params_)
    start = time.time()
    if pred_proba:
        y_pred = grid_search.predict_proba(x_test.values)
    else:
        y_pred = grid_search.predict(x_test.values)
    end = time.time()
    prediction_time = end - start
    if conf.classifier_evaluation_plot:
        perform_classifier_evaluation_plot(classifier_name, x_train, y_train)
    return y_pred, training_time, prediction_time


def perform_classifier_evaluation_plot(classifier_name, x_train, y_train):
    """
    Perform classification evaluation plot
    :param classifier_name: the name of the classifier
    :param x_train: the features of the training set
    :param y_train: the label of the training set
    """
    clf = conf.classifier_models[classifier_name]
    param_name = conf.trade_off_param_name[classifier_name]
    param_range = conf.trade_off_param_range[classifier_name]
    inspector_file_name = conf.current_trade_off_file_name + classifier_name + '.pdf'
    ClassifierTradeOffInspector(clf, x_train, y_train.values.ravel(), classifier_name, file_name=inspector_file_name,
                                param_name=param_name, param_range=param_range)
