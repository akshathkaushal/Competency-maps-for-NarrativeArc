import numpy as np
from gensim.models import FastText

from .word_embedder import WordEmbedderBaseClass


class FasttextEmbedder(WordEmbedderBaseClass):
    def __init__(self, corpus, embedding_size):
        WordEmbedderBaseClass.__init__(self, "FastText", corpus, embedding_size)

    def load_model(self, model_path):
        self.model = FastText.load(model_path)

    def train_model(self, docs=None):
        if docs is not None:
            corpus = docs
        else:
            corpus = self.text

        self.model = FastText(size=self.embedding_size, window=3, min_count=1)
        self.model.build_vocab(sentences=corpus)
        self.model.train(
            sentences=corpus, total_examples=len(corpus), epochs=10
        )

    def get_volume(self, word=None):
        if word is not None:
            return (
                np.abs(np.sum(np.log(np.abs(self.model.wv[word]))))
                if word in self.model.wv
                else 9999
            )
        else:
            return 9999

    def get_embedding(self, word=None):
        if word is not None:
            return self.model.wv[word]
        else:
            return None
