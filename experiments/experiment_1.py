#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import pandas as pd
from experiment import run_experiment

# ratings
ratings_df = pd.read_csv('../data/ratings.tsv', header=None, sep='\t')
ratings_df.rename(columns={0: 'uid', 1: 'movie_id', 2: 'rating', 3: 'timestamp'}, inplace=True)
ratings_df.set_index('uid')
ratings_df.head(5)

run_experiment(ratings_df, 'age')

