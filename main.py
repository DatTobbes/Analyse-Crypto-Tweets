from tweet_analyzer.tweet_online_streamer import OnlineTweetStreamer
from tweet_analyzer.tweet_preprocessor import Preprocessor
import pandas as pd


def read_csv(path):
    return pd.read_csv(path, header=None, names=['time_stamp', 'tweet_text', 'retweeted',
                                                 'retweet_count', 'sentiment_pos',
                                                 'sentiment_neg', 'sentiment_neu', 'senti', 'ment_comp', 'price_diff',
                                                 'start_price', 'end_price'])


FILE_PATH = './data/tweets_test.csv'
BATCHES = 1
BATCH_SIZE = 1000

dataframe = read_csv(FILE_PATH)
tweets = pd.DataFrame(data=dataframe['tweet_text'])

text_preprocessor = Preprocessor()
streamer = OnlineTweetStreamer(text_preprocessor, tweets)

streamer.get_batches(BATCHES, BATCH_SIZE)


text_preprocessor.plot_word_frequency(10)

