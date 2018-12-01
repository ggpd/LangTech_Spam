import os
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

import urllib.request
import tarfile
import shutil

dataset_links = [
        ('ham', 'https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2'),
        ('ham', 'https://spamassassin.apache.org/old/publiccorpus/20021010_hard_ham.tar.bz2'),
        ('spam','https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2'),
        ('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2'),
        ('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'),
        ('ham', 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'),
        ('spam','https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'),
        ('spam','https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2'),
        ('spam','https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'),
]

def load_data(links=dataset_links, data_dir='data', remove_old=True):
    if not os.path.isdir(data_dir) or remove_old:
        if os.path.isdir(data_dir):
            shutil.rmtree(data_dir)
        
        os.mkdir(data_dir)
        download_all_datasets(links, data_dir)

    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    for fol in folders:
        files = [f for f in os.listdir(fol) if os.path.isfile(os.path.join(fol, f))]
        

    


def download_all_datasets(links, data_dir):
    counter = 0
    for ds in links:
        filepath = os.path.join(data_dir,ds[0] + str(counter))
        urllib.request.urlretrieve(ds[1], filepath + '.tar.bz2')
        tar = tarfile.open(filepath + '.tar.bz2', 'r:bz2')
        for member in tar.getmembers():
            if member.isfile():
                tar.extract(member, os.path.join(data_dir,ds[0]))
        tar.close()

        print(filepath + " download finished.")
        counter += 1


def load_all_documents(path, label):
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    documents = []
    for f in files:
        documents.append(load_document(f, label))

    return documents

def load_document(path, label):
    fp = open(path, 'r')
    text = fp.read()
    fp.close()

    return LambDocument(text, label)

def split_documents(documents, train=0.8, rand_seed=1):
    shuffled_docs = shuffle(documents, random_state=rand_seed)
    split_index = train * len(shuffled_docs)

    return shuffled_docs[:split_index], shuffled_docs[split_index:]

def find_feature_words(documents):
    corpus = [d.content for d in documents]

    t = TfidfVectorizer()
    t.fit_transform(corpus)
    return t.get_feature_names()

load_data()
