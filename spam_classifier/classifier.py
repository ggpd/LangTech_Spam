"""
Wrapper class for training model

"""

from sklearn.metrics import confusion_matrix, classification_report
import numpy

from sklearn import model_selection, neighbors
from .plot import plot_confusion_matrix 


class StatelessClassifier:
    """
    Returns the state at the end of each method for easy storage,
    does not store internal data.

    """

    __slots__ = ['train_data', 'train_label']

    def __init__(self, train_data, train_label):
        self.train_data = train_data
        self.train_label = train_label

    def train(self, classifier, grid_search_params=None, train_folds=5):
        """
        Train a classifier with the given data.

        :param train_data training documents
        :param train_label training labels that correspond to data train_data[i] = train_label[i]

        :return model with the best esimator
        """

        print("Training...")
        train_data_matrix = numpy.asarray(self.train_data, dtype=object)
        print("Train data shape: ", train_data_matrix.shape)

        train_label_matrix = numpy.asarray(self.train_label, dtype=object)
        print("Train label shape: ", train_label_matrix.shape)

        cla = None

        if grid_search_params != None:
            classifier = self.grid_search(classifier, grid_search_params, train_folds)
            classifier.fit(self.train_data, self.train_label)
            cla = classifier.best_estimator
        else:
            cla = classifier.fit(self.train_data, self.train_label)

        return cla

    def test(self, classifier, test_data, test_label, output_matrix_png=False):
        """
        Test a classifier with test data.
        """

        test_data_matrix = numpy.asarray(test_data, dtype=object)
        test_label_matrix = numpy.asarray(test_label, dtype=object)

        predictions = classifier.predict(test_data_matrix)
        conf_matrix = confusion_matrix(test_label_matrix, predictions)
        print(classification_report(test_label, predictions))

        if output_matrix_png:
            plot_confusion_matrix(conf_matrix, classifier.classes_, normalize=True, title="Confusion Matrix")

        return predictions
    
    def grid_search(self, classifier, param_grid, nfolds):
        return model_selection.GridSearchCV(classifier, param_grid, cv=nfolds, verbose=1, n_jobs=-1)

