from nltk.corpus import stopwords
from nltk.util import ngrams
import nltk.tokenize
from nltk.stem.porter import PorterStemmer

from bs4 import BeautifulSoup

import email
import re
from collections import Counter

class LambDocument(object):

    __slots__ = ['idd', 'content_soup', 'content', 'label', 'sents', 'words', 'num_word']
    url_regex = '/(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])/igm'


    def __init__(self, idd, text, label):
        email = email.message_from_string(text)
        content = email.get_payload()
        self.text_soup = BeautifulSoup(content, 'html.parser')

        self.id = idd;
        self.label = label
        raw_text  = self.text_soup.get_text()
        self.sents = nltk.tokenize.sent_tokenize(raw_text, language='english')
        self.num_word = len(raw_text)

        # Preprocess text
        raw_text = raw_text.lower() # all lower case
        raw_text = raw_text.translate(None, string.punctuation) # strip puncuation
        raw_text = [w for w in raw_text if not w in stopwords.words('english')] # strip stopwords
        stemmer = PorterStemmer()
        raw_text = [stemmer.stem(w) for w in raw_text] # remove stems

        self.content = raw_text

        self.word_freq = nltk.tokenize.word_tokenize(self.content, language='english')

    def num_sent(self):
        return self.sents.size()

    def avg_sen_len(self):
        return num_word // self.num_sent()

    def ngrams(self, n, top=-1):
        counts = Counter(ngrams(self.word_freq))
        if top <= 0:
            return counts

        return counts.most_common(top)

    def count_links(self):
        return self.text_soup.find_all(href=True).size() # add regex to find all links

    def count_images(self):
        return self.text_soup.find_all('img').size()

    def avg_word_len():
        pass

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
