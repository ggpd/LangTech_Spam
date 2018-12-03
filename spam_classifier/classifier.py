import csv
import os
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy
from sklearn import svm, model_selection, neighbors


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

        train_data_matrix = numpy.asarray(train_data, dtype=object)
        print("Train data shape: ", train_data_matrix.shape)

        train_label_matrix = numpy.asarray(train_label, dtype=numpy.float64)
        print("Train label shape: ", train_label_matrix.shape)

        params = self.grid_search_svc(train_data_matrix, train_label_matrix, 5)

        print(params)

        m = svm.SVC(**params)

        #m = neighbors.KNeighborsClassifier()

        classifier = m.fit(train_data_matrix, train_label_matrix)

        return classifier

    def test(self, classifier, test_data, test_label):
        print("Testing...")

        test_data_matrix = numpy.asarray(test_data, dtype=object)
        test_label_matrix = numpy.asarray(test_label, dtype=numpy.float64)

        predictions = classifier.predict(test_data)
        conf_matrix = confusion_matrix(test_label, predictions)
        print(conf_matrix)
        print(classification_report(test_label, predictions))
        
        return predictions

    def grid_search_randomforest(self, X, y, nfolds):
        pass

    def grid_search_svc(self, X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}
        grid_search = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        return grid_search.best_params_
