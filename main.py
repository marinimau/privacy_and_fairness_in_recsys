#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

from conf import label_names, recs_file_names
from experiment import recs_experiment, observation_experiment


def main():
    """
    main
    :return:
    """
    for label in label_names:
        observation_experiment(label)
        for dataset in recs_file_names:
            recs_experiment(dataset, label)


if __name__ == '__main__':
    main()

