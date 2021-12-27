#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 17/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#


import pandas as pd
import numpy as np

import conf


def balance_data(df, label_name):
    """
    Balance gender data
    :param df: the dataframe
    :param label_name: the current label
    :return: a balanced dataframe
    """
    true_value = 1 if label_name == 'gender' else True
    false_value = 0 if label_name == 'gender' else False
    print("Before rows: " + str(len(df)))
    dff_true = df[df['class'] == true_value]
    dff_false = df[df['class'] == false_value]
    if len(dff_true) > len(dff_false):
        a = dff_true
        b = dff_false
    else:
        a = dff_false
        b = dff_true
    print("# smaller class: " + str(len(b)))
    a = a.sample(n=len(b), random_state=1)
    dff_balanced = pd.concat([a, b]).sample(frac=1, random_state=47)
    print("Balanced rows: " + str(len(dff_balanced)))
    return dff_balanced


def do_temporal_splitting(df):
    """
    Do temporal splitting: maintain the first n records for each user (based on timestamp - asc), n is a percentage
    :param df: the dataframe
    """
    print('Time splitting: ')
    if conf.data_root == conf.data_root_list[0]:
        print('size before: ' + str(df.size))
        grouped = df.sort_values(['timestamp'], ascending=True).groupby('uid')
        df = grouped.apply(lambda x: x.head(int(len(x) * conf.time_split_current_cutoff)))
        print('size after: ' + str(df.size))
    return df


def merge_data(user_df, features_df, on_recs=False):
    """
    Merge features with user labels
    :param user_df: the user dataframe
    :param features_df: the features df
    :param on_recs: a flag that indicates that we are merging recs
    :return:
    """
    if not on_recs:
        features_df = features_df.pivot(index='uid', columns='movie_id', values='rating').fillna(0)
        merged_data = pd.merge(features_df, user_df, on='uid')
    else:
        features_df.drop(columns='rating', inplace=True)
        features_df = features_df.groupby('uid')
        dff = pd.DataFrame()
        for name, group in features_df:
            dff = dff.append(pd.Series([name] + group['movie_id'].tolist()), ignore_index=True)
        dff.rename(columns={0: 'uid'}, inplace=True)
        dff.set_index('uid')
        merged_data = pd.merge(dff, user_df, on='uid')
    merged_data.astype(int)
    merged_data.columns = merged_data.columns.astype(str)
    return merged_data


def merge_embeddings(user_df, features_df):
    """
    Merge embeddings data with user labels
    :param user_df: the user dataframe
    :param features_df: the features df
    :return:
    """
    features_df["uid"] = features_df.index
    # remove inf and fill nan
    features_df.replace([np.inf], 999, inplace=True)
    features_df.replace([-np.inf], -999, inplace=True)
    features_df = features_df.fillna(0)
    features_df = features_df.astype(int)
    # merge features with balanced user
    merged_data = pd.merge(features_df, user_df, on='uid')
    return merged_data
