#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 16/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import conf
import csv


def init_result_file(classifier_name, label_name):
    """
    Init a csv result file
    :param classifier_name: : the name of the classifier
    :param label_name:  the name of the label
    :return:
    """
    file_name = get_file_name(classifier_name, label_name)
    f = open(file_name, "w+")
    f.close()


def write_metrics_row(classifier_name, dataset_name, label_name, metrics):
    """
    Write results row on existing csv file
    :param classifier_name: the name of the classifier
    :param dataset_name: the name of the dataset
    :param label_name: the name of the label
    :param metrics: the metrics score (list: output of evaluator.get_evaluation_metrics)
    :return:
    """
    validate_dataset(dataset_name)
    file_name = get_file_name(classifier_name, label_name)
    row = [dataset_name] + list(metrics)
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def write_metrics(dataset_name, label_name, metrics):
    """
    Write results row on existing csv file
    :param dataset_name: the name of the dataset
    :param label_name: the name of the label
    :param metrics: the metrics score (list: output of evaluator.get_evaluation_metrics)
    :return:
    """
    write_metrics_row(conf.classifiers[0], dataset_name, label_name, metrics[0])
    write_metrics_row(conf.classifiers[1], dataset_name, label_name, metrics[1])


def validate_classifier(classifier_name):
    """
    validate classifier
    :param classifier_name: the name of classifier
    :return:
    """
    assert classifier_name in conf.classifiers


def validate_label(label_name):
    """
    validate label
    :param label_name: the label name
    :return:
    """
    assert label_name in conf.label_names


def validate_dataset(dataset_name):
    """
    validate dataset
    :param dataset_name: the name of the dataset
    :return:
    """
    print(dataset_name)
    assert(dataset_name in conf.dataset_names)


def get_file_name(classifier_name, label_name):
    """
    get the file name given the name of the classifier and the name of the class
    :param classifier_name: the name of the classifier
    :param label_name: the name of the class
    :return:
    """
    validate_classifier(classifier_name)
    validate_label(label_name)
    return str('./results/' + label_name + '_inference_' + classifier_name + '.csv')


