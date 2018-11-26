import click
import os
import time
from . import main, classifier


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


def abs_listdir(data_dir):
    abspaths = list()
    file_list = os.listdir(data_dir)
    for filename in file_list:
        abspath = os.path.abspath(os.path.join(data_dir, filename))
        abspaths.append(abspath)
    return abspaths


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
def data(data_dir, output_dir):
    use_ids = classifier.StatelessClassifier()
    data_paths = abs_listdir(data_dir)
    loaded_data = use_ids.preprocess_data(data_paths)
    train_data, test_data = use_ids.train_test_split(loaded_data)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    use_ids.output_data(train_path, train_data)
    use_ids.output_data(test_path, test_data)


@cli.command()
@click.argument("training_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("classifier_out", type=click.Path(dir_okay=False, file_okay=True))
def train(training_file, classifier_out):
    use_ids = classifier.StatelessClassifier()
    train_data = use_ids.load_preprocessed_data(training_file)
    os.makedirs(os.path.dirname(classifier_out), exist_ok=True)
    final_classifier = use_ids.train(train_data)
    main.output_classifier(classifier_out, final_classifier)


@cli.command()
@click.argument("test_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("classifier_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
def test(test_file, classifier_file):
    use_ids = classifier.StatelessClassifier()
    test_data = use_ids.load_preprocessed_data(test_file)
    final_classifier = main.import_classifier(classifier_file)
    use_ids.test(final_classifier, test_data)


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
def auto(data_dir, output_dir):
    use_ids = classifier.StatelessClassifier()
    data_paths = abs_listdir(data_dir)
    loaded_data = use_ids.preprocess_data(data_paths)

    train_data, test_data = use_ids.train_test_split(loaded_data)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    use_ids.output_data(train_path, train_data)
    use_ids.output_data(test_path, test_data)

    use_ids = classifier.StatelessClassifier()
    train_data = use_ids.load_preprocessed_data(train_path)
    start = time.time()
    final_classifier = use_ids.train(train_data)
    end = time.time()
    print("Time to train: ", end-start)
    classifier_file = os.path.join(output_dir, "classifier.pkl")
    main.output_classifier(classifier_file, final_classifier)

    use_ids = classifier.StatelessClassifier()
    test_data = use_ids.load_preprocessed_data(test_path)
    final_classifier = main.import_classifier(classifier_file)
    start = time.time()
    use_ids.test(final_classifier, test_data)
    end = time.time()
    print("Time to test: ", end-start)
