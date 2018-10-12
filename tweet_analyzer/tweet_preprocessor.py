"""
Mit diesem Script werden Tweets aus einer Tabelle ausgelesen und 
die einzelnen Wörter in den Tweets gezählt
"""

from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob


class Preprocessor:

    def __init__(self):
        self.counter = Counter()
        self.tokenized = []
        self.stop_words = set(stopwords.words('english'))
        self.tokenized_without_most_frequent = []

    def tokenizer(self, text):
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        words = [word for word in tokens if word.isalpha()]
        self.tokenized = [w for w in words if w not in self.stop_words]

        self.tokenized = self.__lemmatize()
        self.counter.update(self.tokenized)
        return self.tokenized

    def __steeming(self):

        porter = PorterStemmer()
        return [porter.stem(word) for word in self.tokenized]

    def __lemmatize(self):

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in self.tokenized]

    def __spelling_Correction(self):
        return [TextBlob(word) for word in self.__lemmatize()]

    def get_word_counts(self, n=10):
        return self.counter.most_common(n)

    def __normalize_word_frequencies(self, tokens, n=10):
        c = Counter()
        c.update(tokens[:n])
        word_freq = list(c.values())
        words = list(c.keys())

        normalized = word_freq / np.max(word_freq)

        return zip(words, normalized, word_freq)

    def get_words_counts_for_tweet(self):
        mapped_tweets = []

        word_counts = sorted(self.counter, key=self.counter.get, reverse=True)

        word_to_int = {word: ii for ii, word in
                       enumerate(word_counts, 1)}

        mapped_tweets.append([word_to_int[word] for word in self.tokenized])

        return mapped_tweets, word_to_int, word_counts

    def remove_most_frequent_words(self, n=10):
        most_frequent = sorted(self.counter, key=self.counter.get, reverse=True)[:n]
        # words = list(self.counter.keys())
        words = self.counter
        self.tokenized_without_most_frequent = [w for w in words if w.keys() not in most_frequent]
        return self.tokenized_without_most_frequent

    def to_Dataframe(self):

        final = self.counter
        df = pd.DataFrame.from_dict(data=final, orient='index').reset_index()
        df = df.rename(columns={'index': 'words', 0: 'count'})
        return df

    def plot_word_frequency(self, n=100):

        df = self.to_Dataframe()
        plt.scatter(x=df['words'][:n], y=df['words'][:n], s=df['count'][:n])

        plt.xticks(np.arange(0, n), x)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.show()
