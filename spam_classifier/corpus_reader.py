from os import listdir
from os.path import isfile, join
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

import urllib.request

dataset_links = [
        ('ham', ''),
        ('spam', ''),
        
]
def download_all_datasets(datasets=dataset_links):
    pass

def load_all_documents(path, label):
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
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
