import os
import csv
import click
from sklearn.externals import joblib
from . import corpus_reader, classifier

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
@click.argument("data_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
def data(data_dir, output_dir):
    print("Loading in data...")
    d = corpus_reader.load_data(data_dir=data_dir)

    train, test = corpus_reader.split_documents(d)

    print("Making vectors...")
    train_v = corpus_reader.get_vectors(train)
    test_v = corpus_reader.get_vectors(test)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    print("Saving vectors...")
    corpus_reader.save_vectors(train_v, train_path)
    corpus_reader.save_vectors(test_v, test_path)

    print("Done processing data")


@cli.command()
@click.argument("training_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("classifier_out", type=click.Path(dir_okay=False, file_okay=True))
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


@cli.command()
@click.argument("test_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("classifier_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("errors_out", type=click.Path(dir_okay=False, file_okay=True))
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


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
def auto(data_dir, output_dir):
    pass


if __name__ == '__main__':
    cli()
