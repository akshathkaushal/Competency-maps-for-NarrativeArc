import configparser
from pathlib import Path


class CompetencyMapConfig:
    """ Class to store the config parameters for generation of the competency map.

    These config parameters can be passed either from a file or if left unspecified will use default values.

    Attributes:
        PREPROCESSOR_TYPE (str): Preprocessing Library to use. spacy or nltk. Defaults to spacy
        TF_IDF_ENABLED (bool): Indicates if tfidf filtering should be performed on the corpus
        FILTER_EXTREMES (bool): Indicates if the filter_extremes function is to be used before topic modeling
        TOPIC_MODEL_TYPE  (str): Type of topic model to Use. lda, ldamallet, hierarchical and top2vec. Defaults to top2vec.
        TOPIC_MODEL_EVALUATION_METRIC (str): Type of Evaluation Metric to use to determine the best topic model.
        Possible values are cv_score, umass_score, lp. Defaults to cv_score.
        NUM_TOPIC_CLUSTERS (int): Maximum Number of Topic Clusters to iterate over. Defaults to 10
        NUM_TOPICS  (int): Number of topics to be extracted from each topic cluster. Defaults to 10
        NUM_LEVELS  (int): Number of Levels to be present on the Y-Axis. Defaults to 10
        WORD_EMBEDDING_TYPE (str): Word Embedding Model to Use. word2vec, glove, fasttext. Defaults to glove
        WORD_EMBEDDING_DIMENSIONS(int): Vector Length for each word embedding. Defaults to 300
        DOCUMENT_EMBEDDING_TYPE  (str): Document Embedding Model to use: doc2vec, pretrained. Defaults to doc2vec
        DOCUMENT_EMBEDDING_DIMENSIONS  (int): Length of each document embedding vector to generate/use. Defaults to 300
        PRETRAINED_DOCUMENT_EMBEDDING_PATH  (str): If pretrained DOCUMENT_EMBEDDING_TYPE is used, then the path to the
        pretrained model. Defaults to None.
    """

    def __init__(self, model_config_path=None):
        self.PREPROCESSOR_TYPE = "spacy"
        self.TF_IDF_ENABLED = True
        self.FILTER_EXTREMES = True
        self.TOPIC_MODEL_TYPE = "hierarchical"
        self.TOPIC_MODEL_EVALUATION_METRIC = "cv_score"

        self.MIN_NUM_TOPIC_CLUSTERS = 2
        self.MAX_NUM_TOPIC_CLUSTERS = -1
        self.NUM_TOPICS = -1
        self.NUM_LEVELS = 10

        self.WORD_EMBEDDING_TYPE = "fasttext"
        self.WORD_EMBEDDING_DIMENSIONS = 300
        self.DOCUMENT_EMBEDDING_TYPE = "doc2vec"
        self.DOCUMENT_EMBEDDING_DIMENSIONS = 300
        self.PRETRAINED_DOCUMENT_EMBEDDING_PATH = None

        if model_config_path is not None:
            # Load the various config values from the input file. If properties are missing, then use the default value.
            import os

            if os.path.exists(model_config_path):
                parameters_config_file = Path(model_config_path)
                config = configparser.ConfigParser()
                config.read(parameters_config_file)
                print(config.sections())
                print(f'Config: {config.get("preprocessor", "type")}')

            if config.has_option("preprocessor", "type"):
                self.PREPROCESSOR_TYPE = config.get("preprocessor", "type")

            if config.has_option("topic_model", "type"):
                self.TOPIC_MODEL_TYPE = config.get("topic_model", "type")

            if config.has_option("topic_model", "enable_tfidf"):
                self.TF_IDF_ENABLED = bool(
                    config.get("topic_model", "enable_tfidf")
                )

            if config.has_option("topic_model", "filter_extremes"):
                self.FILTER_EXTREMES = bool(
                    config.get("topic_model", "filter_extremes")
                )

            if config.has_option("topic_model", "min_num_topic_clusters"):
                self.MIN_NUM_TOPIC_CLUSTERS = config.getint(
                    "topic_model", "min_num_topic_clusters"
                )
            if config.has_option("topic_model", "max_num_topic_clusters"):
                self.MAX_NUM_TOPIC_CLUSTERS = config.getint(
                    "topic_model", "max_num_topic_clusters"
                )

            if config.has_option("topic_model", "evaluation_metric"):
                self.TOPIC_MODEL_EVALUATION_METRIC = config.get(
                    "topic_model", "evaluation_metric"
                )

            if config.has_option("topic_model", "num_topics"):
                self.NUM_TOPICS = config.getint("topic_model", "num_topics")

            if config.has_option("embeddings", "word_embedding_type"):
                self.WORD_EMBEDDING_TYPE = config.get(
                    "embeddings", "word_embedding_type"
                )

            if config.has_option("embeddings", "word_embedding_dimensions"):
                self.WORD_EMBEDDING_DIMENSIONS = config.getint(
                    "embeddings", "word_embedding_dimensions"
                )

            if config.has_option("embeddings", "document_embedding_type"):
                self.DOCUMENT_EMBEDDING_TYPE = config.get(
                    "embeddings", "document_embedding_type"
                )

            if config.has_option("embeddings", "document_embedding_dimensions"):
                self.DOCUMENT_EMBEDDING_DIMENSIONS = config.getint(
                    "embeddings", "document_embedding_dimensions"
                )

            if config.has_option("embeddings", "document_embedding_model_path"):
                self.PRETRAINED_DOCUMENT_EMBEDDING_PATH = config.get(
                    "embeddings", "document_embedding_model_path"
                )

            if config.has_option("embeddings", "num_levels"):
                self.NUM_LEVELS = config.getint("embeddings", "num_levels")
