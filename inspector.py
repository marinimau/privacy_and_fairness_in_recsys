#
#   privacy_and_fairness_in_recsys copyright © 2021 - all rights reserved
#   Created at: 23/11/21
#   By: mauromarini
#   License: MIT
#   Repository: https://github.com/marinimau/privacy_and_fairness_in_recsys
#   Credits: @marinimau (https://github.com/marinimau)
#

import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt

import conf


class ClassifierTradeOffInspector:
    """
    Check trade-off for a given classification method
    """
    __vc_train_score = []
    __vc_val_score = []
    __lc_train_score = []
    __lc_val_score = []

    def __init__(self, classifier, x_train, y_train, experiment_title, param_name, param_range):
        """
        Init
        :param classifier: the classifier
        :param x_train: the training set features
        :param y_train: the training set labels
        :param experiment_title: the title of the experiment
        :param param_name: the name of the param
        :param param_range: a range of values for the param
        """
        self.__classifier = classifier
        self.__x_train = x_train
        self.__y_train = y_train
        self.__experiment_title = experiment_title
        self.__param_range = param_range
        self.__param_name = param_name

        self.__generate_validation_curve()
        self.__generate_learning_curve()
        self.__perform_plot()

    def __generate_validation_curve(self):
        """
        Generate validation curve
        :return:
        """
        self.vc_train_score, self.vc_val_score = model_selection.validation_curve(self.__classifier, self.__x_train,
                                                                                  self.__y_train,
                                                                                  param_name=self.__param_name,
                                                                                  param_range=self.__param_range,
                                                                                  scoring='f1')

    def __generate_learning_curve(self):
        """
        Generate validation curve
        :return:
        """
        _, self.lc_train_score, self.lc_val_score = model_selection.learning_curve(self.__classifier, self.__x_train,
                                                                                   self.__y_train)

    def __perform_plot(self):
        """
        Perform curve plot
        :return:
        """
        # validation curve
        plt.subplot(211)
        plt.title('Validation curve')
        self.__perform_subplot(self.vc_train_score, self.vc_val_score, self.__param_range)
        # learning curve
        plt.subplot(212)
        plt.title('Learning curve')
        print(self.__param_range)
        # self.__perform_subplot(self.lc_train_score, self.lc_val_score, self.__param_range)
        plt.savefig('results/trade-off/' + str(self.__experiment_title) + '.pdf')
        if conf.SHOW_PLOT:
            plt.show()
        plt.close()

    def __perform_subplot(self, train_score, val_score, param_range):
        """
        Perform subplot
        :param train_score: the train score
        :param val_score: the validation score
        :param param_range: the range for the x axis
        :return:
        """
        plt.tight_layout()
        plt.plot(param_range, np.median(train_score, 1), color='blue', label='training score')
        plt.plot(param_range, np.median(val_score, 1), color='red', label='validation score')
        plt.legend(loc='best')
        plt.ylim(0, 1)
        plt.xlim(2, param_range[-1])
        plt.xlabel(self.__param_name)
        plt.ylabel('accuracy')
