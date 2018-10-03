"""
Mit diesem Script werden Tweets aus einer Tabelle ausgelesen und 
die einzelnen Wörter in den Tweets gezählt
"""

import re
import matplotlib.pyplot as plt

from collections import Counter

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


class Preprocessor:

    def __init__(self):
        self.counter = Counter()

    def tokenizer(self, text):
        stop_words = set(stopwords.words('english'))

        # text = re.sub('<\[^>*>', '', text)
        # emoticons = re.findall("(\?::|;|=)(\?:-)?(\?:\)|\(|D|P)", text.lower())
        tokens = word_tokenize(text)
        words = [word for word in tokens if word.isalpha()]
        # text = "{0}{1}".format(re.sub('\[\W+', ' ', text.lower()), ' '.join(emoticons).replace('-', ''))
        tokenized = [w for w in words if w not in stop_words]

        tokenized = self.steeming(tokenized)

        self.counter.update(tokenized)
        return tokenized

    def steeming(self, tokens):
        # stemming of words

        porter = PorterStemmer()

        return [porter.stem(word) for word in tokens]

    def get_word_counts(self, n=100):
        return self.counter.most_common(n)

    def get_words_counts_for_tweet(self, tweet):
        mapped_tweets = []

        word_counts = sorted(self.counter, key=self.counter.get, reverse=True)

        word_to_int = {word: ii for ii, word in
                       enumerate(word_counts, 1)}

        mapped_tweets.append([word_to_int[word] for word in tweet.split()])

