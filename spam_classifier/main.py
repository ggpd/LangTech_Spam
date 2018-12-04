import os
import csv
from sklearn.externals import joblib
from . import corpus_reader, classifier


def data(data_dir, output_dir):
    print("Loading in data...")
    d = corpus_reader.load_data(data_dir=data_dir)

    train, test = corpus_reader.split_documents(d)

    print("Making vectors...")
    features = corpus_reader.find_feature_words(train, 50)
    train_v = corpus_reader.get_vectors(train, features)
    test_v = corpus_reader.get_vectors(test, features)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    print("Saving vectors...")
    corpus_reader.save_vectors(train_v, train_path)
    corpus_reader.save_vectors(test_v, test_path)

    print("Done processing data")
    return train_path, test_path


def train(training_file, classifier_out):
    train_vectors = corpus_reader.load_vectors(training_file)
    train_label = [x[1] for x in train_vectors]
    train_data = [x[2:] for x in train_vectors]
    #train_data = train_data[:10]
    #train_label = train_label[:10]
    #print(train_label)

    use_ids = classifier.StatelessClassifier()

    final_classifier = use_ids.train(train_data, train_label)

    joblib.dump(final_classifier, filename=classifier_out)
    return classifier_out


def test(test_file, classifier_file, errors_out):
    test_csv = corpus_reader.load_vectors(test_file)
    test_label = [x[1] for x in test_csv]
    test_file_names = [x[0] for x in test_csv]
    test_data = [x[2:] for x in test_csv]

    use_ids = classifier.StatelessClassifier()

    final_classifier = joblib.load(filename=classifier_file)

    pred = use_ids.test(final_classifier, test_data, test_label)

    with open(errors_out, 'w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for label, p, fname in zip(test_label, pred, test_file_names):
            if label != p:
                writer.writerow([label, p, fname])
