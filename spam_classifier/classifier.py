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
    def preprocess_data(self, data_paths):
        combined_values = list()
        combined_classes = list()
        for path in data_paths:
            # TODO Rework for spam dataset
            pass
        return combined_values, combined_classes

    def output_data(self, path, data):
        values, classes = data
        print("Writing data to: ", path)
        with open(path, 'w') as out_file:
            writer = csv.writer(out_file, quoting=csv.QUOTE_NONNUMERIC)
            for value, output_class in zip(values, classes):
                writer.writerow(value + [output_class])
        print("Done writing data to: ", path)

    def load_preprocessed_data(self, path):
        values = list()
        classes = list()
        with open(path, 'r') as in_file:
            data = csv.reader(in_file, quoting=csv.QUOTE_NONNUMERIC)
            for data_item in data:
                values.append(data_item[:-1])
                classes.append(data_item[-1])
        return values, classes

    def train_test_split(self, data, random_state=1, test_size=0.2):
        values, classes = data
        train_values, test_values, train_classes, test_classes = train_test_split(values, classes, random_state=random_state, test_size=test_size)
        return (train_values, train_classes), (test_values, test_classes)

    def train(self, train_data):
        train_values, train_classes = train_data
        print("Training...")
        train_classes = numpy.asarray(train_classes, dtype=object)
        print("Train class shape: ", train_classes.shape)
        train_values = numpy.asarray(train_values, dtype=object)
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
