#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

from sklearn.model_selection import train_test_split


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
