from abc import ABCMeta, abstractmethod


class WordEmbedderBaseClass(metaclass=ABCMeta):
    def __init__(self, name=None, corpus=None, embedding_size=150):
        self.model_path = None
        self.name = None
        self.text = None
        if name:
            self.name = name
        if corpus is not None:
            self.text = corpus
        self.model = None
        self.embedding_size = embedding_size

    @abstractmethod
    def train_model(self, docs=None):
        pass

    @abstractmethod
    def get_volume(self, word=None):
        pass

    @abstractmethod
    def get_embedding(self, word=None):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass
