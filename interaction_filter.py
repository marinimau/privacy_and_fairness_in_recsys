#
#   privacy_and_fairness_in_recsys copyright © 2022 - all rights reserved
#   Created at: 11/01/22
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#



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
