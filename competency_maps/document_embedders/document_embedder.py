from abc import ABCMeta, abstractmethod


class DocumentEmbedderBaseClass(metaclass=ABCMeta):
    """  Metaclass for defining Document Embedder Model
    Use this metaclass for defining a Document Embedder Model. A Document Embedder Model Class is used to obtain the
    resource embeddings and subsequently the resource volumes for each learning resource provided as input.
    Attributes:
        name (str): Document Embedding Model Type
        corpus (DataFrame): Corpus to be used for training the model.
        model: Document Embedding Model obtained after trainning or loaded from an existing path.
    """

    def __init__(self, name=None, corpus=None, embedding_size=150):
        """ Initialise the class attributes
        Args:
            name (str): Document Embedding Model type.
            corpus (DataFrame): Training Data to be used.
        """
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
    def get_volume(self, document=None):
        pass

    @abstractmethod
    def get_embedding(self, document=None):
        pass

    @abstractmethod
    def load_model(self, model_path):
        pass
