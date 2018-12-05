from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy

from sklearn import svm, model_selection, neighbors
from .plot import plot_confusion_matrix 



class StatelessClassifier(object):
    """
    Returns the state at the end of each method for easy storage,
    does not store internal data.

    """

    def train(self, train_data, train_label):
        print("Training...")

        train_data_matrix = numpy.asarray(train_data, dtype=object)
        print("Train data shape: ", train_data_matrix.shape)

        train_label_matrix = numpy.asarray(train_label, dtype=object)
        print("Train label shape: ", train_label_matrix.shape)

        # grid_search = self.grid_search_svc(train_data_matrix, train_label_matrix, 5)
        # grid_search = self.grid_search_knn(train_data_matrix, train_label_matrix, 5)
        # grid_search = self.grid_search_nb(train_data_matrix, train_label_matrix, 5)
        grid_search = self.grid_search_rf(train_data_matrix, train_label_matrix, 5)

        print(grid_search.best_params_)
        # print(grid_search.best_estimator_.feature_importances_)
        return grid_search.best_estimator_

    def test(self, classifier, test_data, test_label):
        print("Testing...")

        test_data_matrix = numpy.asarray(test_data, dtype=object)
        test_label_matrix = numpy.asarray(test_label, dtype=object)

        predictions = classifier.predict(test_data_matrix)
        conf_matrix = confusion_matrix(test_label_matrix, predictions)
        print(conf_matrix)
        print(classification_report(test_label, predictions))

        plot_confusion_matrix(conf_matrix, ['ham', 'spam'], normalize=True)

        return predictions

    def grid_search_rf(self, X, y, nfolds):
        estimators = [5, 10, 20, 40]
        max_depths = [5, 10, 20, 40, None]
        max_features = ["auto", "log2", None]
        param_grid = {'n_estimators': estimators, 'max_depth': max_depths, 'max_features': max_features}
        grid_search = self.grid_search(RandomForestClassifier(random_state=1), param_grid, nfolds)
        grid_search.fit(X, y)
        return grid_search

    def grid_search_svc(self, X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = self.grid_search(svm.SVC(kernel='rbf'), param_grid, nfolds)
        grid_search.fit(X, y)
        return grid_search

    def grid_search_nb(self, X, y, nfolds):
        fit_prior = [True, False]
        smoothing = [1.0e-10, 1.0]
        param_grid = {'fit_prior': fit_prior, 'alpha': smoothing}
        grid_search = self.grid_search(BernoulliNB(), param_grid, nfolds)
        grid_search.fit(X, y)
        return grid_search

    def grid_search_knn(self, X, y, nfolds):
        n_neighbors = [1, 3, 5, 10, 15, 20]
        weights = ['uniform', 'distance']
        param_grid = {'n_neighbors': n_neighbors, 'weights': weights}
        grid_search = self.grid_search(KNeighborsClassifier(), param_grid, nfolds)
        grid_search.fit(X, y)
        return grid_search

    def grid_search(self, classifier, param_grid, nfolds):
        return model_selection.GridSearchCV(classifier, param_grid, cv=nfolds, verbose=1, n_jobs=-1)

