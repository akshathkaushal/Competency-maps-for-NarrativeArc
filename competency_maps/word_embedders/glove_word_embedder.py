import numpy as np

from glove import Corpus, Glove

from .word_embedder import WordEmbedderBaseClass


class GloveEmbedder(WordEmbedderBaseClass):
    def __init__(self, corpus, embedding_size):
        WordEmbedderBaseClass.__init__(self, "Glove", corpus, embedding_size)
        self.corpus_model = None

    def train_model(self, docs=None):
        if docs is None:
            corpus = self.text
        else:
            corpus = docs

        self.corpus_model = Corpus()
        self.corpus_model.fit(corpus, window=10)
        self.model = Glove(
            no_components=self.embedding_size, learning_rate=0.05
        )
        self.model.fit(self.corpus_model.matrix, verbose=True)
        self.model.add_dictionary(self.corpus_model.dictionary)

    def load_model(self, model_path):
        self.model = Glove.load(model_path)

    def get_volume(self, word=None):
        if word is not None:
            return (
                np.abs(
                    np.sum(
                        np.log(
                            np.abs(
                                self.model.word_vectors[
                                    self.model.dictionary[word]
                                ]
                            )
                        )
                    )
                )
                if word in self.model.dictionary.keys()
                else 9999
            )
        else:
            raise ValueError("Word not specified..!!")

    def get_embedding(self, word=None):
        if word is not None:
            return self.model.word_vectors[self.model.dictionary[word]]
        else:
            raise ValueError("Word not specified..!!")
