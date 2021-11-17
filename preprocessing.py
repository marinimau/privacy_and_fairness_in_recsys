#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 17/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#


import pandas as pd


def balance_data(df):
    """
    Balance gender data
    :param df: the dataframe
    :return: a balanced dataframe
    """
    print("Before rows: " + str(len(df)))
    dff_true = df[df['class'] == 1]
    true_count = len(dff_true)
    print("True rows: " + str(true_count))
    dff_false = df[df['class'] == 0].sample(n=true_count, random_state=1)
    dff_balanced = pd.concat([dff_true, dff_false]).sample(frac=1, random_state=47)
    print("Balanced rows: " + str(len(dff_balanced)))
    return dff_balanced
