from tweet_analyzer.tweet_online_streamer import OnlineTweetStreamer
from tweet_analyzer.tweet_preprocessor import Preprocessor
import pandas as pd


def read_csv(path):
    return pd.read_csv(path, header=None, names=['time_stamp', 'tweet_text', 'retweeted',
                                                 'retweet_count', 'sentiment_pos',
                                                 'sentiment_neg', 'sentiment_neu', 'senti', 'ment_comp', 'price_diff',
                                                 'start_price', 'end_price'])


FILE_PATH = './data/tweets_test.csv'
BATCHES = 3
BATCH_SIZE = 100

dataframe = read_csv(FILE_PATH)
tweets = pd.DataFrame(data=dataframe['tweet_text'])

text_analyzer = Preprocessor()
streamer = OnlineTweetStreamer(text_analyzer, tweets)

streamer.get_batches(BATCHES, BATCH_SIZE)
#word_freq = text_analyzer.remove_most_frequent_words()
#print(word_freq)

#mapped_tweets, word_to_int, word_counts = text_analyzer.get_words_counts_for_tweet()
#print(mapped_tweets)
#print(word_counts)
#print(word_to_int)

text_analyzer.plot_word_frequency(100)
text_analyzer.remove_most_frequent_words(10)