"""
Extractor of text features.

"""

import re
import string
from collections import Counter
import mailparser
from functools import reduce

from nltk.corpus import stopwords
import nltk.tokenize
from nltk.stem.porter import PorterStemmer

from bs4 import BeautifulSoup


class LambDocument(object):
    """
    A LambDocument is a document that hasn't been classified, and to be used to
    extract features for classification.
    """

    __slots__ = [
        'idd',
        'text',
        'text_soup',
        'label',
        'sents',
        'tokens',
        'title_tokens',
        'word_freq',
        'title_word_freq',
        'num_attach',
        'num_word',
        'avg_word_len'
    ]

    def __init__(self, idd, text, label):
        mail = mailparser.parse_from_string(text)
        self.text = mail.body
        self.num_attach = len(mail.attachments)
        self.text_soup = BeautifulSoup(self.text, 'html.parser')

        self.idd = idd
        self.label = label
        raw_text = self.text_soup.get_text()
        self.sents = nltk.tokenize.sent_tokenize(raw_text, language='english')
        self.num_word = len(raw_text)

        # Preprocess text
        trans = str.maketrans('', '', string.punctuation)

        raw_text = raw_text.lower().translate(trans) # strip puncuation
        self.tokens = nltk.tokenize.word_tokenize(raw_text, language='english')

        title = mail.subject.lower().translate(trans)
        self.title_tokens = nltk.tokenize.word_tokenize(title, language='english')
        
        self.avg_word_len = reduce(lambda x, y: x + len(y), self.tokens, 0) / (len(self.tokens) + 1)
        
        self.tokens = [w for w in self.tokens if not w in stopwords.words('english')]
        stemmer = PorterStemmer()
        self.tokens = [stemmer.stem(w) for w in self.tokens] # remove stems

        self.title_tokens = [w for w in self.title_tokens if not w in stopwords.words('english')]
        self.title_tokens = [stemmer.stem(w) for w in self.title_tokens] # remove stems

        self.word_freq = Counter(self.tokens)
        self.title_word_freq = Counter(self.title_tokens)

    def num_sent(self):
        """ 
        Get the total number of sentences. 
        
        :return number of sentences
        """
        return len(self.sents)

    def avg_sen_len(self):
        """
        Get the average sentence length.

        :return average sentence length
        """
        return self.num_word // (1+self.num_sent())

    def count_links(self):
        """
        Count the amount of links based on HTML href.

        :return amount of links
        """
        return len(self.text_soup.find_all(href=True))

    def count_images(self):
        """
        Count the amount of images based on HTML img tag.

        :return amount of images
        """

        return len(self.text_soup.find_all('img'))

    def word_freq_select(self, word_select):
        """
        For all words in word_select, get frequency and return in vector.
        0 if not found.

        :param word_select words to find

        :return vector of lengths
        """
        return [self.word_freq[w] for w in word_select]

    def title_word_freq_select(self, word_select):
        """
        For all words in word_select, get frequency and return in vector.
        0 if not found.

        :param word_select words to find

        :return vector of lengths
        """
        return [self.title_word_freq[w] for w in word_select]


    def get_vector(self, word_select_features, title_word_select_features):
        """
        Get a vector from document features.

        """
        total_capital, per_len_capital = count_capital(self.text)
        vec = [
            self.idd,
            self.label,
            self.num_word,
            self.avg_word_len,
            self.avg_sen_len(),
            self.count_links(),
            self.count_images(),
            self.num_attach,
            total_capital
        ]

        vec += per_len_capital
        vec += self.title_word_freq_select(title_word_select_features)
        vec += self.word_freq_select(word_select_features)

        return vec

    def get_vector_description(self, word_select_features, title_select_features, cap_seq_min=3, cap_seq_max=10):
        return ["num_words",
                "avg_word_len",
                "avg_sentence_len",
                "count_links",
                "count_images",
                "number_attachments",
                "total_capitals",
                *["capital_seq_" + str(x) for x in range(cap_seq_min, cap_seq_max + 1)],
                *title_select_features,
                *word_select_features]


def count_capital(text, min_track=3, max_track=10):
    """
    Get the amount of capital letters, along with a distribution of capital letter
    sequences.

    :param text the text to process
    :param min_track min sequence to track
    :param max_track max sequence to track

    :return amount of capital letters
    :return tuple of sequence length to frequency.
    """

    if max_track < min_track:
        return None

    count = 0
    capital_freq = {}
    current_streak = 0
    for char in text:
        if char.isupper():
            count += 1
            current_streak += 1
        else:
            if current_streak < min_track and min_track > 0:
                current_streak = 0
                continue

            if current_streak > max_track and max_track > 0:
                current_streak = 0
                continue

            if current_streak not in capital_freq:
                capital_freq[current_streak] = 0

            capital_freq[current_streak] = capital_freq[current_streak] + 1
            current_streak = 0

    if current_streak >= min_track and \
        min_track > 0 and \
        current_streak <= max_track and \
        max_track > 0:

        if current_streak not in capital_freq:
            capital_freq[current_streak] = 0

        capital_freq[current_streak] = capital_freq[current_streak] + 1

    return count, [(x, capital_freq[x]) if x in capital_freq else (x, 0) 
            for x in range(min_track, max_track+1)]
