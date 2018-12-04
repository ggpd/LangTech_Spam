import os

import click

from spam_classifier import main

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


@cli.command()
@click.argument("data_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
def data(data_dir, output_dir):
    main.data(data_dir, output_dir)


@cli.command()
@click.argument("training_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("classifier_out", type=click.Path(dir_okay=False, file_okay=True))
def train(training_file, classifier_out):
    main.train(training_file, classifier_out)


@cli.command()
@click.argument("test_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("classifier_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.argument("errors_out", type=click.Path(dir_okay=False, file_okay=True))
def test(test_file, classifier_file, errors_out):
    main.test(test_file, classifier_file, errors_out)


@cli.command()
@click.argument("data_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
def auto(data_dir, output_dir):
    classifier_out = os.path.join(output_dir, "classifier.pkl")
    errors_out = os.path.join(output_dir, "errors.csv")
    train_out, test_out = main.data(data_dir, output_dir)
    main.train(train_out, classifier_out)
    main.test(test_out, classifier_out, errors_out)


@cli.command()
@click.argument("train_file", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("test_file", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
def train_test(train_file, test_file, output_dir):
    classifier_out = os.path.join(output_dir, "classifier.pkl")
    errors_out = os.path.join(output_dir, "errors.csv")
    main.train(train_file, classifier_out)
    main.test(test_file, classifier_out, errors_out)

if __name__ == '__main__':
    cli()
