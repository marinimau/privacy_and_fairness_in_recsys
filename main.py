#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

from conf import label_names, recs_file_names
from experiment import experiment_automation


def main():
    """
    main
    :return:
    """
    for label in label_names:
        for dataset in recs_file_names:
            experiment_automation(dataset, label)


if __name__ == '__main__':
    main()

