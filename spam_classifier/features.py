from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk.tokenize
from nltk.stem.porter import PorterStemmer

from bs4 import BeautifulSoup

import string
from collections import Counter
import mailparser
from functools import reduce


class LambDocument(object):

    __slots__ = [
            'idd', 
            'text_soup', 
            'label', 
            'sents', 
            'tokens', 
            'words', 
            'num_attach', 
            'num_word', 
            'avg_word_len'
            ]

    def __init__(self, idd, text, label):
        mail = mailparser.parse_from_string(text)
        text = mail.body
        self.num_attach = len(mail.attachments)
        self.text_soup = BeautifulSoup(text, 'html.parser')

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
        self.avg_word_len = reduce(lambda x,y: x + len(y), self.tokens, 0) / (len(self.tokens) + 1)
        self.tokens = [w for w in self.tokens if not w in stopwords.words('english')] # strip stopwords
        stemmer = PorterStemmer()
        self.tokens = [stemmer.stem(w) for w in self.tokens] # remove stems

        self.words = Counter(self.tokens)

    def num_sent(self):
        return len(self.sents)

    def avg_sen_len(self):
        return self.num_word // (1+self.num_sent())

    def ngrams(self, n, top=-1):
        counts = Counter(ngrams(self.word_freq, n))
        if top < 0:
            return counts

        return counts.most_common(top)

    def count_links(self):
        return len(self.text_soup.find_all(href=True))

    def count_images(self):
        return len(self.text_soup.find_all('img'))

    def word_freq(self, word_select):
        freqs = []

        for w in word_select:
            freqs.append(self.words[w])
        return freqs

    def get_vector(self, word_features):
        vec = [
                self.idd, 
                self.label, 
                self.num_word, 
                self.avg_word_len,
                self.avg_sen_len(),
                self.count_links(), 
                self.count_images(), 
                self.num_attach
                ]
        vec += self.word_freq(word_features)

        return vec


def count_capital(text):
    count = 0
    capital_freq = {}
    current_streak = 0
    for char in text:
        if char.isupper():
            count += 1
            current_streak += 1
        else:
            if current_streak > 0:
                if current_streak in capital_freq:
                    capital_freq[current_streak] = 0 

                capital_freq[current_streak] = capital_freq[current_streak] + 1
                current_streak = 0

    return count, [(k,v) for k,v in capital_freq.iteritems()]
