import os
import csv
import math
import tarfile
import shutil
import urllib.request

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from .features import LambDocument
from .plot import plot_wordcloud 

import nltk

dataset_links = [
    ('ham', 'https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2'),
    #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2'),
    ('spam', 'https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2'),
    #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2'),
    #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'),
    #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'),
    #('spam', 'https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'),
    #('spam', 'https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2'),
    #('spam', 'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'),
]


def setup_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

def get_vectors(data, features):
    return [d.get_vector(features) for d in data]

def save_vectors(vectors, filepath):
    with open(filepath, 'w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for v in vectors:
            writer.writerow(v)

def load_vectors(filename):
    vectors = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for v in reader:
            vectors.append(v)
    return vectors

def load_data(links=dataset_links, data_dir='data', remove_old=False):
    if not os.path.isdir(data_dir) or remove_old:
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
 
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, 'spam'))
        os.mkdir(os.path.join(data_dir, 'ham'))

        download_all_datasets(links, data_dir)

    data = []
    folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for fol in folders:
        files = [os.path.join(fol,f) for f in os.listdir(fol) if os.path.isfile(os.path.join(fol, f))]
        for file_number, fil in enumerate(files):
            with open(fil, errors='replace', encoding='utf-8') as f:
                print(file_number, " ", f.name)
                d = f.read()
                try:
                    doc = LambDocument(
                            os.path.basename(f.name),
                            d,
                            os.path.basename(os.path.dirname(fil))
                            )
                    data.append(doc)
                except AssertionError:
                    print("Skipping file #", file_number, ", path: ", f.name)
                    print("Encoding error")

    return data


def download_all_datasets(links, data_dir):
    counter = 0
    for ds in links:
        filepath = os.path.join(data_dir,ds[0] + str(counter))
        urllib.request.urlretrieve(ds[1], filepath + '.tar.bz2')
        tar = tarfile.open(filepath + '.tar.bz2', 'r:bz2')
        for member in tar.getmembers():
            if member.isfile():
                f = tar.extractfile(member)
                fwrite = open(os.path.join(data_dir, ds[0], os.path.basename(member.name)), 'wb+')
                fwrite.write(f.read())

                fwrite.close()
                f.close()
        tar.close()

        print(filepath + " download finished.")
        counter += 1

def split_documents(documents, train=0.8, rand_seed=1):
    train, test = train_test_split(documents, train_size=train, random_state=rand_seed)
    return train, test

def find_feature_words(documents, feature=50, export_graph=True):
    spam_corpus = [' '.join(d.tokens) for d in documents if d.label == 'spam']
    ham_corpus = [' '.join(d.tokens) for d in documents if d.label == 'ham']

    t = TfidfVectorizer(max_features=feature)

    t.fit_transform(spam_corpus)
    spam_features = t.get_feature_names()
    if export_graph:
        plot_wordcloud(spam_features, spam_corpus, "spam.png")

    t.fit_transform(ham_corpus)
    ham_features = t.get_feature_names()
    if export_graph:
        plot_wordcloud(ham_features, ham_corpus, "ham.png")

    return spam_features + ham_features
