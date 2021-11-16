#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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
        'max_features': ['auto', 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train.values.ravel())
    y_pred = grid_search.predict(x_test)
    return y_pred


def perform_logistic_regression(x_train, y_train, x_test):
    """
    Make predictions with LogisticRegression
    :param x_train: the x of training set
    :param y_train: the y of the training set (label)
    :param x_test: the x of the test set
    :return:
    """
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred
