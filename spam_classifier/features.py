from  nltk.corpus import stopwords
from nltk.util import ngrams
import nltk.tokenize
from collections import Counter
from bs4 import BeautifulSoup
from functools import reduce

class LambDocument(object):

    __slots__ = ['text_soup', 'text', 'label', 'sents', 'words', 'num_word']

    def __init__(self, text, label):
        self.text_soup = BeautifulSoup(text, 'html.parser')
        self.label = label
        self.text = self.text_soup.get_text()
        self.sents = nltk.tokenize.sent_tokenize(text, language='english')

        no_punc_words = lowers.lower().translate(None, string.punctuation)
        num_word = no_punc_words.len()

        filter_stop = [w for w in no_punc_words if not w in stopwords.words('english')]
        self.word_freq = nltk.tokenize.word_tokenize(no_punc_words, language='english')

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
