#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 08/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import zipfile
import io
import requests
import os
import pandas as pd

from conf import urls, VERBOSE


def extract_data(response, names):
    """
    Extract data from the response and generate tsv files.
    :param response: the downloaded file
    :param names: the names of the files contained in the dataset
    :return:
    """
    for name in names:
        data = []
        print(f"extracting {name}.dat..." if VERBOSE else "")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            for line in zip_ref.open(f"ml-1m/{name}.dat"):
                data.append(str(line, "latin-1").replace("::", "\t"))
        os.makedirs("data", exist_ok=True)
        with open(f"data/{name}.tsv", "w") as f:
            f.writelines(data)
        print("done." if VERBOSE else "")


def perform_loading():
    """
    Perform data loading
    :return:
    """
    print("download data..." if VERBOSE else "")
    response = requests.get(urls["movielens1m"])
    print("done." if VERBOSE else "")
    extract_data(response, ["ratings", "users", "movies"])


def get_users():
    """
    Returns users df
    :return: the user dataframe
    """
    user_df = pd.read_csv('./data/users.tsv', header=None, sep='\t')
    user_age_df = user_df.iloc[:, : 3]
    user_age_df.rename(columns={0: 'uid', 1: 'gender', 2: 'age'}, inplace=True)
    user_age_df.set_index('uid')
    return user_age_df


def get_age_labels():
    """
    Returns the user age labels
    :return: The user age labels
    """
    user_df = get_users()
    user_age_df = user_df.drop('gender', 1)
    user_age_df.rename(columns={'age': 'class'}, inplace=True)
    return user_age_df


def get_gender_labels():
    """
    Returns the user gender labels
    :return: The user gender labels
    """
    user_df = get_users()
    user_gender_df = user_df.drop('age', 1)
    user_gender_df.rename(columns={'gender': 'class'}, inplace=True)
    user_gender_df['class'].replace('F', 0, inplace=True)
    user_gender_df['class'].replace('M', 1, inplace=True)
    return user_gender_df


def get_best_recs(relevance, k=10):
    """
    Given the relevance list get the best k items for each user
    :param relevance: the relevance list
    :param k: the number of best recs to take
    :return: a df with k-recs for each user
    """
    return relevance.groupby('uid').head(k)
