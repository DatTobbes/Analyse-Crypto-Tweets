"""
This Class analyze the tweets row by row
"""
import pyprind
import pandas as pd


class OnlineTweetStreamer:

    def __init__(self, preprocessor, tweets):
        self.cleaned_text = []
        self.dataframe = tweets
        self.row_iterator = self.dataframe.iterrows()
        self.preprocessor = preprocessor
        self.row_counter= 0

    def stream(self):
        co = 0
        for i, row in self.row_iterator:
            self.row_counter +=1
            text = self.preprocessor.tokenizer(row['tweet_text'])
            yield text

    def get_minibatch(self, size):

        docs = []
        try:
            for _ in range(size):
                text = next(self.stream())
                docs.append(text)

        except StopIteration:
            return None

        return docs

    def get_batches(self, batches=1, batch_size=10):

        pbar = pyprind.ProgBar(batches,
                               title='Counting words')
        for _ in range(batches):
            pbar.update()
            self.cleaned_text= self.get_minibatch(batch_size)

        print('{:d} rows read from csv'.format(self.row_counter))
