## A module to create word embeddings for the training data using pre-train Glove and Word2Vec embeddings.
## I've imported the module and run all models. Unfortunately all the cross-validation scores are below 0.6
## so my final result was based on tfidf.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api


class WordEmbedding(object):
    def __init__(self, data, emb_source):
        self.data = data
        self.source = emb_source


    def gen_embed(self,weight):
        max_len = np.max([len(sent) for sent in self.data])
        embedding = np.zeros((7917, max_len))
        if self.source == 'word2vec':
            pre_trained = api.load('word2vec-google-news-300')
        elif self.source == 'glove':
            pre_trained = api.load('glove-wiki-gigaword-300')

        tfidf = TfidfVectorizer(norm='l2', lowercase=False).fit(self.data)
        for rid, sent in enumerate(self.data):
            for wid, word in enumerate(sent.split()):
                if word in pre_trained:
                    if weight == "mean":
                        embedding[rid, wid] = np.mean(pre_trained.get_vector(word), axis=0)
                    if weight == 'sum':
                        embedding[rid, wid] = np.sum(pre_trained.get_vector(word), axis=0)
                    if weight == 'tfidf':
                        idf_weights = 0
                        if word in tfidf.vocabulary_.keys():
                            idf_weights = tfidf.idf_[tfidf.vocabulary_[word]]
                        embedding[rid, wid] = np.mean(pre_trained.get_vector(word) * idf_weights, axis=0)

        return embedding















