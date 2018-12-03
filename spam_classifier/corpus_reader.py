import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import urllib.request
import tarfile
import shutil
from .features import LambDocument

import nltk
from collections import Counter
import csv

dataset_links = [
        ('ham', 'https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2'),
        #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2'),
        ('spam','https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2'),
        #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2'),
        #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'),
        #('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'),
        #('spam','https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'),
        #('spam','https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2'),
        #('spam','https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'),
]


def setup_resources():
    nltk.download('punkt')
    nltk.download('stopwords')


def get_vectors(data):
    features = find_feature_words(data, feature=500)
    return [d.get_vector(features) for d in data]


def save_vectors(vectors, filepath="vectors.csv"):
    with open(filepath, 'w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for v in vectors:
            writer.writerow(v)


def load_vectors(filename="vectors.csv"):
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
        for fil in files:
            with open(fil, errors='replace') as f:
                print(str(len(data)) + " " + f.name)
                d = f.read()
                doc = LambDocument(
                        os.path.basename(f.name), 
                        d, 
                        os.path.basename(os.path.dirname(fil))
                        )
                data.append(doc)

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


def find_feature_words(documents, feature=50):
    corpus = [" ".join(d.tokens) for d in documents]

    t = TfidfVectorizer(max_features=feature)
    t.fit_transform(corpus)
    return t.get_feature_names()


def find_top_words(documents):
    tokens = []
    for doc in documents:
        tokens += doc.tokens

    counter = Counter(tokens)

    return counter

