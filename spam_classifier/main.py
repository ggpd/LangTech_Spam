from sklearn.externals import joblib


def output_classifier(path, classifier):
    joblib.dump(classifier, filename=path, compress=True)


def import_classifier(path):
    classifier = joblib.load(filename=path)
    return classifier
