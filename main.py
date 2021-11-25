#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import conf
from experiment import recs_experiment, observation_experiment
from utils import get_inspector_file_name


def main():
    """
    main
    :return:
    """
    for label in conf.label_names:
        conf.current_trade_off_file_name = get_inspector_file_name('observation', label)
        observation_experiment(label)
        for dataset in conf.recs_file_names:
            conf.current_trade_off_file_name = get_inspector_file_name(dataset, label)
            recs_experiment(dataset, label)


if __name__ == '__main__':
    main()

