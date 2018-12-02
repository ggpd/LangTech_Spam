import csv
import os
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy


def _fix_nulls(s):
    for line in s:
        yield line.replace('\0', '')


class StatelessClassifier:
    """
    Returns the state at the end of each method for easy storage,
    does not store internal data.

    """

    def train(self, train_data, train_label):
        print("Training...")

        train_classes = numpy.asarray(train_classes, dtype=object)
        print("Train class shape: ", train_classes.shape)

        train_values = numpy.asarray(train_labels, dtype=object)
        print("Train value shape: ", train_values.shape)

        classifier = MLPClassifier(hidden_layer_sizes=(20,))

        classifier.fit(train_values, train_classes)
        return classifier

    def test(self, classifier, test_data):
        test_values, test_classes = test_data
        print("Testing...")
        predictions = classifier.predict(test_values)
        conf_matrix = confusion_matrix(test_classes, predictions)
        print(conf_matrix)
        print(classification_report(test_classes, predictions))
