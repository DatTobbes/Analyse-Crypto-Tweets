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
        self.tokenized =[]
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

    def __get_word_counts(self, n=10):
        return self.counter.most_common(n)

    def __normalize_word_frequencies(self, n=10):

        word_freq = []
        words = []
        for k,v in self.__get_word_counts(n):
            words.append(k)
            word_freq.append(v)

        mean_values = np.mean(word_freq)
        std_values = np.std(word_freq)

        normalized = word_freq/np.max(word_freq)

        return zip(words, normalized, word_freq)

    def get_words_counts_for_tweet(self):
        mapped_tweets = []

        word_counts = sorted(self.counter, key=self.counter.get, reverse=True)

        word_to_int = {word: ii for ii, word in
                       enumerate(word_counts, 1)}

        mapped_tweets.append([word_to_int[word] for word in self.tokenized])

        return mapped_tweets, word_to_int, word_counts

    def remove_most_frequent_words(self, n=10):

        lst = self.counter.most_common(n)
        df = pd.DataFrame(lst, columns=['Word', 'Count'])

        most_frequent = self.counter.most_common(n)
        self.tokenized_without_most_frequent= [w for w in self.tokenized if w not in most_frequent]
        return self.tokenized_without_most_frequent

    def plot_word_frequency(self, n=100):

        norm_set = self.__normalize_word_frequencies(n)

        x, y, s = zip(*norm_set)
        plt.scatter(x=x, y=y, s=s)

        plt.xticks(np.arange(0,n, step= n/10),x)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.show()