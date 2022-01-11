#
#   privacy_and_fairness_in_recsys copyright © 2022 - all rights reserved
#   Created at: 11/01/22
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import conf
import os

import pandas as pd


def group_split(x):
    """
    Perform the splitting given the user interactions
    :param x: the user interactions
    :return: the filtered group
    """
    user_interactions = x
    subset_size = int(len(user_interactions) / conf.n_subset_for_filtering)
    subsets = user_interactions.iloc[0:subset_size - 1]
    for i in range(1, conf.n_subset_for_filtering - 1):
        subsets.append(user_interactions.iloc[i * subset_size:(i + 1) * subset_size - 1].sample(
            frac=conf.filtering_sampling_percentages[i], random_state=698))
    return subsets


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
        print('size before: ' + str(len(original_data)))
        grouped = original_data.sort_values(['timestamp'], ascending=False).groupby('uid')
        original_data = grouped.apply(group_split)
        print('size after: ' + str(len(original_data)))
    return original_data


def load_original():
    """
    load original data
    :return: the original interactions
    """
    original_df = pd.read_csv('data/' + conf.data_root + '/ratings.tsv', header=None, sep='\t')
    original_df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating', 3: 'timestamp'}, inplace=True)
    original_df.set_index('uid')
    return original_df


def save_data(filtered_data):
    """
    save filtered data in a .tsv file
    :param filtered_data: the filtered data
    """
    path = 'data_obfuscated/' + conf.data_root + '/' + conf.obfuscation_path[2]
    os.makedirs(path, exist_ok=True)
    filtered_data.to_csv(path + 'ratings.tsv', sep='\t', header=False)


save_data(interaction_filter(load_original()))
