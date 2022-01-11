#
#   privacy_and_fairness_in_recsys copyright © 2022 - all rights reserved
#   Created at: 11/01/22
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import conf

import pandas as pd


def group_split(x):
    """
    Perform the splitting given the user interactions
    :param x: the user interactions
    :return: the filtered group
    """
    return x.head(1)


def interaction_filter(original_data):
    """
    Time filtering on the interactions data:
    - order by timestamp desc
    - group by user
    - split each group in 4 equal subset,
    - take respectively the 100%, 80%, 60%, 40% of interaction from each subset (100% in the subset that contains
    latest interactions)
    :param original_data: the interactions original data
    :return: the filtered data
    """
    if conf.data_root == conf.data_root_list[0]:
        print('Interaction filter: ')
        print('size before: ' + str(original_data.shape))
        grouped = original_data.sort_values(['timestamp'], ascending=False).groupby('uid')
        original_data = grouped.apply(group_split)
        print('size after: ' + str(original_data.size))
    return original_data


def load_original():
    """
    load original data
    :return: the original interactions
    """
    original_df = pd.read_csv('data/' + conf.data_root + '/ratings.tsv', header=None, sep='\t')
    original_df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating', 3: 'timestamp'}, inplace=True)
    original_df.set_index('uid')
    print(len(original_df))
    return original_df#.drop_duplicates(subset=['uid', 'movie_id'])


interaction_filter(load_original())
