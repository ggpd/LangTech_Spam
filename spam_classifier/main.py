"""
Implementation of main functions of the classifier.

"""

import os
import csv
from sklearn.externals import joblib
from spam_classifier import corpus_reader, classifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

rf_param_grid = {
    'n_estimators': [5, 10, 20, 40], 
    'max_depth': [5, 10, 20, 40, None], 
    'max_features': ['auto', 'log2', None]
}

svc_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10], 
    'gamma': [0.001, 0.01, 0.1, 1]
}

nb_param_grid = {
    'fit_prior': [True, False], 
    'alpha': [1.0e-10, 1.0]
}

knn_param_grid = {
    'n_neighbors': [1, 3, 5, 10, 15, 20], 
    'weights': ['uniform', 'distance']
}

def data(data_dir, output_dir):
    """
    Load data, create vectors and save to output directory.

    :param data_dir directory to save data
    :param output_dir directory to output vectors

    :return train_path name of file the training data saved too
    :return test_path name of file the test data saved too
    """

    print("Loading in data...")
    d = corpus_reader.load_data(data_dir=data_dir)

    train, test = corpus_reader.split_documents(d)

    print("Making vectors...")
    body_features = corpus_reader.find_feature_words(train, num_feature=50)
    title_features = corpus_reader.find_feature_words(train, num_feature=50, token_parameter='title_word_freq')
    train_v = corpus_reader.get_vectors(train, body_features, title_features)
    test_v = corpus_reader.get_vectors(test, body_features, title_features)

    data_description = train[0].get_vector_description(body_features, title_features)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    print("Saving vectors...")
    corpus_reader.save_vectors(train_v, train_path)
    corpus_reader.save_vectors(test_v, test_path)

    print("Done processing data")
    print(data_description)
    return train_path, test_path


def train(training_file, classifier_out, classifier_str):
    """
    Train and output a model.

    :param training_file training csv
    :param classifier_out file to save classifier too
    :param classifier_str classifier string

    :return trained classifier, None if str is invalid
    """

    cla = None
    param_grid = None

    if classifier_str.lower() == 'svc':
        cla = SVC(kernel='rbf')
        param_grid = svc_param_grid
    elif classifier_str.lower() == 'rf':
        cla = RandomForestClassifier(random_state=1) 
        param_grid = rf_param_grid
    elif classifier_str.lower() == 'knn':
        cla = KNeighborsClassifier()
        param_grid = knn_param_grid
    elif classifier_str.lower() == 'nb':
        cla = BernoulliNB()
        param_grid = nb_param_grid
    else:
        return

    train_vectors = corpus_reader.load_vectors(training_file)
    train_label = [x[1] for x in train_vectors]
    train_data = [x[2:] for x in train_vectors]

    use_ids = classifier.StatelessClassifier(train_data, train_label)

    final_classifier = use_ids.train(cla, grid_search_params=param_grid)

    joblib.dump(final_classifier, filename=classifier_out)
    return classifier_out


def test(test_file, classifier_file, errors_out):
    """
    Test test set on outputted model.

    :param test_file csv of test data
    :param classifier_file classifier file to test
    :param errors_out file to output errors too
    """

    test_csv = corpus_reader.load_vectors(test_file)
    test_label = [x[1] for x in test_csv]
    test_file_names = [x[0] for x in test_csv]
    test_data = [x[2:] for x in test_csv]

    use_ids = classifier.StatelessClassifier(None, None)

    final_classifier = joblib.load(filename=classifier_file)

    pred = use_ids.test(final_classifier, test_data, test_label)

    with open(errors_out, 'w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for label, p, fname in zip(test_label, pred, test_file_names):
            if label != p:
                writer.writerow([label, p, fname])
