"""
Graphing functions for evaluation.

"""

import itertools
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import nltk.tokenize

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', filename='conf_matrix.png', cmap=plt.cm.Reds):
    """
    Plot a confusion matrix with matplotlib.

    :cm confusion matrix
    :classes classes of graph as list
    :normalize normalize to 0-1 scale
    :title title of graph
    :filename filename to save too
    :cmap color map to use
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)

def plot_wordcloud(freq, corpus, filename):
    """
    Create a wordcloud from the frequently used words.

    :freq list of words to extract
    :corpus list of all documents
    :filename name of file to save
    """

    tokens = []
    
    for doc in corpus:
        tokens += nltk.tokenize.word_tokenize(doc, language='english')

    count = Counter(tokens)

    freq_map = {v:float(count[v]) for v in freq}

    wc = WordCloud(
        background_color='white',
        width=2500,
        height=2000
    )

    wc.generate_from_frequencies(freq_map)

    wc.to_file(filename)
