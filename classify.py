#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import numpy as np
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import conf
from inspector import ClassifierTradeOffInspector


def split_data(dff, test_size=0.3, random_state=698):
    """
    Split data for classification phase
    :param dff: the original dataframe
    :param test_size: the percentage for test set splitting
    :param random_state: the random state
    :return:
    """
    x = dff.iloc[:, dff.columns != 'class']
    y = dff.iloc[:, dff.columns == 'class']
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def perform_random_forest(x_train, y_train, x_test):
    """
    Make predictions with Random Forest
    :param x_train: the x of training set
    :param y_train: the y of the training set (label)
    :param x_test: the x of the test set
    :return:
    """
    clf = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True)
    param_grid = {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'n_jobs': [-1],
        'min_samples_leaf': [4, 40, 100, 200],
        'random_state': [101, 698]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
    start = time.time()
    grid_search.fit(x_train.values, y_train.values.ravel())
    end = time.time()
    training_time = end - start
    if conf.VERBOSE:
        print(grid_search.best_params_)
    start = time.time()
    y_pred = grid_search.predict(x_test.values)
    end = time.time()
    prediction_time = end - start
    if conf.classifier_evaluation_plot:
        inspector_file_name = conf.current_trade_off_file_name + 'random_forest.pdf'
        best_clf = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                          max_features=grid_search.best_params_['max_features'])
        c = ClassifierTradeOffInspector(best_clf, x_train, y_train.values.ravel(), "prova",
                                        file_name=inspector_file_name,
                                        param_name='n_estimators',
                                        param_range=np.arange(200, 700, 100))
        del c
    return y_pred, training_time, prediction_time


def perform_logistic_regression(x_train, y_train, x_test):
    """
    Make predictions with LogisticRegression
    :param x_train: the x of training set
    :param y_train: the y of the training set (label)
    :param x_test: the x of the test set
    :return:
    """
    clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=10000, random_state=1)
    start = time.time()
    clf.fit(x_train.values, y_train.values.ravel())
    end = time.time()
    training_time = end - start
    start = time.time()
    y_pred = clf.predict(x_test.values)
    end = time.time()
    prediction_time = end - start
    param_range = [0.001, 0.05, 0.1, 0.5, 1.0, 10.0]
    if conf.classifier_evaluation_plot:
        inspector_file_name = conf.current_trade_off_file_name + 'logistic_regression.pdf'
        c = ClassifierTradeOffInspector(LogisticRegression(solver='lbfgs', max_iter=300), x_train,
                                        y_train.values.ravel(), "prova_l",
                                        param_name='C', param_range=param_range, file_name=inspector_file_name)
        del c
    return y_pred, training_time, prediction_time
