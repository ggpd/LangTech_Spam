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

    __slots__ = [
        'idd',
        'text',
        'text_soup',
        'label',
        'sents',
        'tokens',
        'word_freq',
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
        raw_text = raw_text.lower() # all lower case
        trans = str.maketrans('', '', string.punctuation)

        raw_text = raw_text.translate(trans) # strip puncuation
        self.tokens = nltk.tokenize.word_tokenize(raw_text, language='english')
        self.avg_word_len = reduce(lambda x, y: x + len(y), self.tokens, 0) / (len(self.tokens) + 1)
        self.tokens = [w for w in self.tokens if not w in stopwords.words('english')]
        stemmer = PorterStemmer()
        self.tokens = [stemmer.stem(w) for w in self.tokens] # remove stems

        self.word_freq = Counter(self.tokens)

    def num_sent(self):
        """ Get the total number of sentences. """
        return len(self.sents)

    def avg_sen_len(self):
        return self.num_word // (1+self.num_sent())

    def count_links(self):
        return len(self.text_soup.find_all(href=True))

    def count_images(self):
        return len(self.text_soup.find_all('img'))

    def word_freq_select(self, word_select):
        return [self.word_freq[w] for w in word_select]

    def get_vector(self, word_select_features):
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

        #vec += per_len_capital
        vec += self.word_freq_select(word_select_features)

        return vec

    def get_vector_description(self, word_select_features):
        return ["num_words",
                "avg_word_len",
                "avg_sentence_len",
                "count_links",
                "count_images",
                "number_attachments",
                "total_capitals",
                *word_select_features]


def count_capital(text, min_track=3, max_track=10):
    count = 0
    capital_freq = {}
    current_streak = 0
    for char in text:
        if char.isupper():
            count += 1
            current_streak += 1
        else:
            if current_streak > 0:
                if current_streak < min_track and min_track > 0:
                    current_streak = min_track - 1

                if current_streak > max_track and max_track > 0:
                    current_streak = max_track + 1

                if current_streak not in capital_freq:
                    capital_freq[current_streak] = 0

                capital_freq[current_streak] = capital_freq[current_streak] + 1
                current_streak = 0

    return count, [(k, v) for k, v in capital_freq.items()].sort(key=lambda x: x[0])
