#
#   privacy_and_fairness_in_recsys copyright © 2022 - all rights reserved
#   Created at: 19/01/22
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import pandas as pd

import conf
from experiment import get_metrics_from_classifier
from preprocessing import balance_data
from utils import write_metrics


def main():
    """
    main function
    :return
    """
    models = ['user-knn', 'bprmf', 'wrmf', 'multidae']
    label = ['gender', 'age']
    cutoffs = [10, 20, 50]
    for label in label:
        for model in models:
            for cutoff in cutoffs:
                file_path = './output/altered_' + model + '_' + label + '_cutoff_' + str(cutoff) + '.csv'
                classification_dataset = model + '_classification_best_' + str(cutoff)
                fairrecsys_recs_df = pd.read_csv(file_path, header=0)
                fairrecsys_recs_df.rename(columns={'userIndex': 'uid', 'b': 'class'}, inplace=True)
                if conf.balance_data:
                    fairrecsys_recs_df = balance_data(fairrecsys_recs_df, label)
                metrics = get_metrics_from_classifier(fairrecsys_recs_df, label, already_merged=True)
                write_metrics(label_name=label, dataset_name=classification_dataset, metrics=metrics)


if __name__ == '__main__':
    main()
