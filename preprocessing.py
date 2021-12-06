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

from sklearn.preprocessing import normalize


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


def merge_data(user_df, features_df, on_recs=False):
    """
    Merge data
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
    return normalize(merged_data)


def merge_embeddings(user_df, features_df):
    """
    Merge embeddings data
    :param user_df: the user dataframe
    :param features_df: the features df
    :return:
    """
    features_df.replace([np.inf], 9, inplace=True)
    features_df.replace([-np.inf], -9, inplace=True)
    features_df = features_df.fillna(0)
    merged_data = pd.merge(features_df, user_df, left_index=True, right_index=True)
    merged_data.columns = merged_data.columns.astype(str)
    return merged_data
