import gensim
import numpy as np

from competency_maps.document_embedders.document_embedder import (
    DocumentEmbedderBaseClass,
)


class Doc2vecEmbedder(DocumentEmbedderBaseClass):
    """ Doc2Vec Document Embedding Model Class.

    Doc2Vec is a paragraph embedding model proposed by Quoc Le and Tomas Mikolov in their work
    "Distributed Representations of Sentences and Documents"
    Link: https://arxiv.org/pdf/1405.4053v2.pdf

    We make use of the Gensim Implementation of the Doc2Vec Model here

    Attributes:
        name (str): Document Embedding Model Type
        corpus (DataFrame): Corpus to be used for training the model.
        model: Document Embedding Model obtained after trainning or loaded from an existing path.
    """

    def __init__(self, corpus, embedding_size):
        DocumentEmbedderBaseClass.__init__(
            self, "Doc2Vec", corpus, embedding_size
        )

    def get_embedding(self, document=None):
        """ Obtain resource embedding vector for a given resource

        Args:
            document (list): List of Strings Containing the tokens for the learning resource

        Returns:
            A array containing the embedding vector for the document
        """
        self.model.random.seed(42)
        doc_vector = self.model.infer_vector(document)
        return doc_vector

    def get_volume(self, document=None):
        """ Obtain the resource volume for a given learning resource

        Resource Volume is obtained by taking absolute log sum of the resource embedding vector

        Args:
            document (list): List of Strings Containing the tokens for the learning resource

        Returns:
            A float that holds the resource volume for the specified learning resource
        """
        self.model.random.seed(42)
        doc_vector = self.model.infer_vector(document)
        volume = np.abs(np.sum(np.log(np.abs(doc_vector))))
        return volume

    def train_model(self, docs=None):
        """ Train a Doc2Vec Model using the corpus as input

        Once the model is trained, the model attribute is updated with the trained model obtained here.

        Args:
            docs (list): List of TaggedDocuments that contain the corpus to be used for training
        """
        if docs is None:
            corpus = self.text
        else:
            corpus = docs
        documents = [
            gensim.models.doc2vec.TaggedDocument(doc, [i])
            for i, doc in enumerate(corpus)
        ]
        self.model = gensim.models.Doc2Vec(
            documents,
            vector_size=self.embedding_size,
            window=10,
            min_count=2,
            workers=6,
        )

    def load_model(self, model_path):
        """Load a pretrained doc2vec model to the model attribute from the model_path argument provided as input"""
        self.model = gensim.models.Doc2Vec.load(model_path)
