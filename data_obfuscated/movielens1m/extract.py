#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 30/12/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import pandas as pd


def extract():
    """
    Extract data from the response and generate tsv files.
    :return:
    """
    data = pd.read_csv("./raw/ratings_obfuscated_pred.dat", sep='::', header=None, skipinitialspace=True)
    data.dropna(inplace=True)
    print(data.head())
    data.to_csv('ratings.tsv', sep='\t', index=False, header=None)


extract()
