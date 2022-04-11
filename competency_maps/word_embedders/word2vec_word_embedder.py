import gensim
import numpy as np

from .word_embedder import WordEmbedderBaseClass


class Word2vecEmbedder(WordEmbedderBaseClass):
    def __init__(self, corpus, embedding_size):
        WordEmbedderBaseClass.__init__(self, "Word2vec", corpus, embedding_size)

    def train_model(self, docs=None):
        if docs is None:
            corpus = self.text
        else:
            corpus = docs
        self.model = gensim.models.Word2Vec(
            corpus, size=self.embedding_size, window=10, min_count=2, workers=6
        )

    def load_model(self, model_path):
        self.model = gensim.models.Word2Vec.load(model_path)

    def get_volume(self, word=None):
        if word is not None:
            return (
                np.abs(np.sum(np.log(np.abs(self.model.wv[word]))))
                if word in self.model.wv
                else 9999
            )
        else:
            raise ValueError("Word not specified..!!")

    def get_embedding(self, word=None):
        if word is not None:
            return self.model.wv[word] if word in self.model.wv else []
        else:
            raise ValueError("Word not specified..!!")
