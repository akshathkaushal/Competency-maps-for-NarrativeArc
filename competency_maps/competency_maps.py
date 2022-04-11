"""Main module."""
import logging
import math
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from rich.console import Console
from rich.progress import Progress

from competency_maps import (
    document_embedders,
    preprocessors,
    topic_models,
    word_embedders,
)
from competency_maps.exceptions import exceptions
from competency_maps.utils import custom_logger, dataframe_utility
from competency_maps.utils.map_config import CompetencyMapConfig

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


console = Console()


class CompetencyMap:
    """Class to represent the Competency Map Module and its functionalities

    Competency Map organises a learning space in terms of basic units of learning, each of which is called a competency.
    The competency map is organised as a 2-dimensional progression space. A progression space not only has a concept of
    distance between any pairs of competencies, but also a partial ordering that indicates progress made by a learner
    on reaching some competency. In addition to the idea of progression, a competency map needs to be organised such
    that, learning resources mapped onto the space can create coherent sequences of learning pathways that can be
    traversed on the space.

    Attributes:
        id (str): Unique ID for the map to be generated
        map_config (class::CompetencyMapConfig): Contains the model parameters to be used for building the map
        resources_path (str): Path where the input resources are located
        results_path (str): Path where the results/models will be stored
        delimiter (str, optional): Specifies the delimiter in the file if input is a csv file. Defaults to None
        content_fields (list, optional): Contains the field names that store the content for the map if
         input is a csv file. Defaults to None
        prefix (bool, optional): Indicates the variant used for creating the map. Defaults to False
        is_debug(bool, optional): Indicates if logging of debug messages are required or not. Defaults to True
    """

    def __init__(
        self,
        id,
        map_config,
        corpus,
        results_path,
        content_fields=None,
        is_debug=True,
    ):
        """Initialize the competency map object with required parameters"""
        self.map_config = map_config
        self.map_id = id
        self.corpus = corpus
        self.map_path = Path(f"{results_path}/{self.map_id}")
        self.model_directory = Path(f"{results_path}/{self.map_id}/models")
        self.is_debug = is_debug
        if content_fields is None:
            self.content_fields = ["description"]
        else:
            self.content_fields = content_fields
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        logger_name = f"{self.map_id}_cmap_generation_{timestamp}.log"
        self.logger = custom_logger.my_custom_logger(
            logger_name,
            Path.joinpath(self.map_path, "logs"),
            logging.DEBUG if self.is_debug else logging.INFO,
        )

    def get_clean_corpus(self, corpus):
        """Preprocess the Corpus by using standard preprcoessing techniques

        Creates an object of :class:`competency_maps.preprocessors.Preprocessor`.

        Args:
            corpus: Raw Data frame

        Returns:
            A  class `pandas.DataFrame` with the preprocessd tokens present in tokens field of the dataframe

        """
        corpus["full_text"] = ""
        console.log(f"Content: {self.content_fields}")
        if (
            len(self.content_fields) == 1
            and self.content_fields == "description"
        ):
            corpus["full_text"] = corpus["description"]
        else:
            for field in self.content_fields:
                corpus["full_text"] = (
                    corpus[field].map(str) + ". " + corpus["full_text"].map(str)
                )

        preprocessor = preprocessors.grab(self.map_config.PREPROCESSOR_TYPE)
        corpus["clean_text"] = preprocessor.normalize_corpus(
            corpus["full_text"]
        )
        corpus["tokens"] = preprocessor.get_tokens(corpus["clean_text"])
        return corpus

    def get_topic_model(self, tokens, create=True):
        """ Use the tokens to build a topic model

        Args:
            tokens: Corpus of Text that is preprocessed and converted to tokens
            create: Boolean Indicating whether new map is to be created or not. Defaults to true

        Returns:
            Object of type :class:`competency_maps:competency_maps.topic_models.TopicModel`.

        Raises:
            InvalidTopicModelException: Invalid Topic Model Specified as Input
        """
        topic_model_type = self.map_config.TOPIC_MODEL_TYPE
        topic_model_path = Path.joinpath(
            self.model_directory, f"best_{topic_model_type}_topic_model.obj"
        )
        dictionary_path = Path.joinpath(
            self.model_directory,
            f"best_{topic_model_type}_topic_model_dictionary.obj",
        )
        topic_modeller = topic_models.grab(
            topic_model_type,
            text_corpus=tokens,
            min_topic_clusters=self.map_config.MIN_NUM_TOPIC_CLUSTERS,
            max_topic_clusters=self.map_config.MAX_NUM_TOPIC_CLUSTERS,
            enable_tfidf=self.map_config.TF_IDF_ENABLED,
            enable_filter=self.map_config.FILTER_EXTREMES,
        )
        if os.path.exists(topic_model_path):
            topic_modeller.load_model(
                topic_model_path.as_posix(), dictionary_path.as_posix()
            )
        elif not os.path.exists(topic_model_path) and create:
            # output_results = self.model_directory if self.is_debug else None
            topic_modeller.get_best_model(
                output_dir=self.model_directory,
                metric=self.map_config.TOPIC_MODEL_EVALUATION_METRIC,
            )
        else:
            raise exceptions.InvalidTopicModelException()
        return topic_modeller

    def get_topics(self, topic_model, n_terms):
        """ Obtain List of Topics/Keywords from each topic cluster in the topic model

        Args:
            topic_model: Topic Model from where the terms have to be extracted
            n_terms: Number of terms to be extracted from each cluster.

        Returns:
            :class:`pandas:pandas.DataFrame`
        """
        return topic_model.get_topics(n_terms)

    def get_word_embedding_model(self, tokens, create=True):
        """ Train a Word Embedding Model for the given set of input tokens

        Args:
            tokens: Corpus from which the word embedding model will be trained
            create: Boolean Indicating whether new map is to be created or not. Defaults to true

        Returns:
            Object of Type `WordEmbedder`

        Raises:
            InvalidWordEmbeddingModelException: Invalid Word Embedding Model Specified
        """
        word_embedder_type = self.map_config.WORD_EMBEDDING_TYPE
        word_embedder_model_path = Path.joinpath(
            self.model_directory,
            f"word_embedding_model_{word_embedder_type}.bin",
        ).as_posix()
        word_embedder = word_embedders.grab(
            word_embedder_type,
            corpus=tokens,
            embedding_size=self.map_config.WORD_EMBEDDING_DIMENSIONS,
        )
        if os.path.exists(word_embedder_model_path):
            word_embedder.load_model(word_embedder_model_path)
        elif not os.path.exists(word_embedder_model_path) and create:
            word_embedder.train_model()
            word_embedder.model.save(word_embedder_model_path)
        else:
            raise exceptions.InvalidWordEmbeddingModelException()
        return word_embedder

    def get_topic_volumes(self, topics, word_embedding_model):
        """ Compute the Volume of each topic identified in the topic model using the word embedding model.

        Args:
            topics: List of terms for which volumes have to be computed
            word_embedding_model: Word Embedding Model to be used for computing volumes.

        Returns:
            Dataframe with a new field `topic_volume` that holds the volume for each topic.
        """
        topics["topic_volume"] = topics["topic_name"].apply(
            word_embedding_model.get_volume
        )
        unavailable_topics = topics[topics["topic_volume"] == 9999]
        if len(unavailable_topics) > 0:
            console.log(
                f"Could not Compute Volume for {len(unavailable_topics['topic_name'].unique())} topics."
            )
            console.log(
                f"The unavailable topics are: {unavailable_topics['topic_name'].values}"
            )
        topics = topics[topics["topic_volume"] != 9999]
        return topics

    def get_resource_mapping(self, corpus, topic_model, topics):
        """ Map a Resource to a corresponding Topic Cluster and in turn to all the topics in the cluster

        Args:
            corpus: Dataframe where each row is a resource
            topic_model: Topic Model to be used for mapping a resource to a topic_cluster
            topics: Topics to be fetched from each cluster once the respource is mapped.

        Returns:
            Dataframe with the below fields:

        """
        topic_distributions = topic_model.topic_mappings()
        console.log(topic_distributions)
        corpus["topic_distribution"] = topic_distributions
        exploded_data = dataframe_utility.explode(
            corpus, "topic_distribution", "topic_distribution"
        )
        exploded_data[["topic_id", "topic_score"]] = exploded_data[
            "topic_distribution"
        ].apply(pd.Series)
        joined_df = pd.merge(
            exploded_data,
            topics,
            left_on="topic_id",
            right_on="topic_cluster_id",
        )
        joined_df["document_mapped_probability"] = (
            joined_df["topic_score"] * joined_df["topic_cluster_probability"]
        )
        joined_df.drop(
            ["description", "full_text", "clean_text", "tokens"], axis=1
        ).to_csv(
            Path.joinpath(self.model_directory, "resource_mapping_raw.csv"),
            index=False,
            header=True,
        )
        resource_mapping = joined_df.drop(
            [
                "topic_distribution",
                "topic_id",
                "tokens",
                "topic_score",
                "topic_cluster_probability",
                "topic_cluster_id",
            ],
            axis=1,
        )
        return resource_mapping

    def get_document_embedding_model(self, tokens, create=True):
        """ Obtain a Document Embedding Model

        Args:
            tokens: Text using which the document embedding model will be trained.
            create: Boolean Indicating whether new map is to be created or not. Defaults to true

        Returns:
            document embedding model trained on the corpus of learning resources.

        Raises:
            InvalidDocumentEmbeddingModelException: Invalid Document Embedding Model Specified.
        """
        document_embedder_type = self.map_config.DOCUMENT_EMBEDDING_TYPE
        if self.map_config.PRETRAINED_DOCUMENT_EMBEDDING_PATH is not None:
            document_embedder_type = "doc2vec"
            document_embedder_model_path = (
                self.map_config.PRETRAINED_DOCUMENT_EMBEDDING_PATH
            )
            if not os.path.exists(document_embedder_model_path):
                document_embedder_model_path = Path.joinpath(
                    self.model_directory,
                    "resource_embedding_model_{}.bin".format(
                        document_embedder_type
                    ),
                ).as_posix()
        else:
            document_embedder_model_path = Path.joinpath(
                self.model_directory,
                "resource_embedding_model_{}.bin".format(
                    document_embedder_type
                ),
            ).as_posix()
        document_embedder = document_embedders.grab(
            document_embedder_type,
            corpus=tokens,
            embedding_size=self.map_config.DOCUMENT_EMBEDDING_DIMENSIONS,
        )
        if os.path.exists(document_embedder_model_path):
            document_embedder.load_model(document_embedder_model_path)
        elif not os.path.exists(document_embedder_model_path) and create:
            document_embedder.train_model()
            document_embedder.model.save(document_embedder_model_path)
        else:
            raise exceptions.InvalidDocumentEmbeddingModelException()
        return document_embedder

    @staticmethod
    def get_resource_volumes(corpus, document_embedding_model):
        """ Compute Volume for each document/resource in the corpus

        Args:
            corpus: Dataframe where each row represents a resource
            document_embedding_model: Document Embedding Model from which the volume of a resource will be computed.

        Returns:
            Dataframe with new field `resource_volume` holding the volume for each resource.
        """
        corpus["resource_volume"] = corpus["tokens"].apply(
            document_embedding_model.get_volume
        )
        return corpus

    @staticmethod
    def map_resource_to_best_topic(corpus):
        """
        Map a resource to a single point in the topic volume space. Since a resource is mapped to multiple topics,
        we find the best point on the space to hold the resource. This is done by computing a weighted sum of the
        topic volumes of a resource by weighing it on the resource_topic_probability.

        Args:
            corpus: Input Corpus

        Returns:
            DataFrame with 2 new Fields: X and Y where
                1. X - Best X Coordinate of the resource
                2. Y - Best Y Coordinate of the resource
        """
        map = corpus.sort_values("document_mapped_probability", ascending=False)
        map["norm_prob"] = map["document_mapped_probability"] / map.groupby(
            "resource_id"
        )["document_mapped_probability"].transform("sum")
        map["norm_x"] = map["norm_prob"] * map["topic_volume"]
        map["norm_y"] = map["norm_prob"] * map["resource_volume"]
        weighted_coordinates = map.groupby("resource_id").agg(
            {"norm_x": "sum", "norm_y": "sum"}
        )
        weighted_coordinates.columns = ["X", "Y"]
        raw_map = pd.merge(
            weighted_coordinates,
            map,
            how="right",
            left_on=["resource_id"],
            right_on=["resource_id"],
        )
        return raw_map

    @staticmethod
    def get_intervals(volume_list, n_levels):
        """ Used to convert the volume space into a grid space.

        Args:
            volume_list: List of Volumes
            n_levels: Number of intervals to be obtained

        Returns:
            Min Value and the Interval Length.
        """
        v_min = math.floor(volume_list.min())
        v_max = math.ceil(volume_list.max())
        interval = (v_max - v_min) / n_levels
        return v_min, interval

    def create_map(self):
        """ Method to Create a new Competency Map given a set of input corpus.

        Creation of the competency map consists of the below steps:

        1. Scan the input path and prepare a dataframe where each row represents a learning resource.
        2. Apply preprocessing techniques to each learning resource content
        3. Build a topic model using the cleaned text as input
        4. Train a Word Embedding Model using the cleaned text as input
        5. Train a Document Embedding Model using the cleaned text as input
        6. For each learning resource, compute the resource volume from the resource embedding vector.
        7. Extract Topics from Topics Clusters Identified in Step 3.
        8. Compute Topic Volumes for each topic using the Word Embedding Model
        9. Map each learning resources to a set of topics along with their probability as weights
        10. Obtain the suitable location for each resource by computing a weighted sum of the mapped topics
        11. Scale the values obtained in Step 6 and 10 to the competency map space.

        Returns: Returns the below objects

            1. A dataframe consisting of all the topics identified.
            2. A dataframe consisting of the mapping between the resources and the topics
            3. A dataframe consisting of the resource locations on the map.
            4. A JSON Object containing some properties of the map.
        """
        map_details = {}
        start = time.time()
        console.log("Cleaning Corpus(1/10).")
        preprocessed_texts_path = Path.joinpath(
            self.map_path, "preprocessed_texts.obj"
        )
        if os.path.exists(preprocessed_texts_path):
            clean_corpus = pd.read_pickle(preprocessed_texts_path)
        else:
            clean_corpus = self.get_clean_corpus(self.corpus)
            pd.to_pickle(clean_corpus, preprocessed_texts_path)
        console.log(f"Found {len(clean_corpus)} Learning Resources.!!")
        console.log("Get Topic Model from Cleaned Corpus(2/10)..")
        topic_model = self.get_topic_model(clean_corpus["tokens"])
        console.log(
            f"Generated a Topic Model with {topic_model.best_model_clusters} Clusters.."
        )

        with Progress() as progress:
            task = progress.add_task(
                total=10, description="Building Competency Map"
            )
            progress.advance(task)
            progress.advance(task)
            console.log("Get Word Embedding Model from Cleaned Corpus(3/10)...")
            word_embedding_model = self.get_word_embedding_model(
                clean_corpus["tokens"]
            )
            progress.advance(task)
            console.log(
                "Get Resource Embedding Model from Cleaned Corpus(4/10)...."
            )
            document_embedding_model = self.get_document_embedding_model(
                clean_corpus["tokens"]
            )
            progress.advance(task)
            console.log("Get Resource Volume for each resource(5/10).....")
            corpus_with_resource_volumes = self.get_resource_volumes(
                clean_corpus, document_embedding_model
            )
            progress.advance(task)
            # obtain Topics from topic clusters
            console.log("Get Topics from each topic cluster(6/10)......")
            topics = self.get_topics(topic_model, self.map_config.NUM_TOPICS)
            console.log(
                f"Extracted {len(topics['topic_name'].unique())} Topics from the Topic Clusters!!"
            )
            progress.advance(task)
            console.log("Get Topic Volumes for each topic(7/10).......")
            topic_volumes = self.get_topic_volumes(topics, word_embedding_model)
            progress.advance(task)
            # Map Resources to Topics
            console.log("Map Each Resource to set of topics(8/10)........")
            resource_topic_mapping = self.get_resource_mapping(
                corpus_with_resource_volumes, topic_model, topic_volumes
            )
            progress.advance(task)
            # Obtain Best X-Mapping for X
            console.log("Get Best Mapping for each resource(9/10).........")
            resource_mapping_with_best_topic = self.map_resource_to_best_topic(
                resource_topic_mapping
            )
            c_map = resource_mapping_with_best_topic[
                ["resource_id", "X", "Y"]
            ].drop_duplicates()
            progress.advance(task)
            # Build Competency Map Space
            console.log("Build Competency Map(10/10)..........")
            x_min, intervals_x = self.get_intervals(
                topic_volumes["topic_volume"],
                len(topic_volumes["topic_name"].unique()),
            )
            c_map["norm_X"] = c_map["X"].apply(
                lambda x: (x - x_min) / intervals_x
            )
            y_min, intervals_y = self.get_intervals(
                c_map["Y"], self.map_config.NUM_LEVELS
            )
            c_map["norm_Y"] = c_map["Y"].apply(
                lambda y: (y - y_min) / intervals_y
            )
            final_map = c_map[["resource_id", "norm_X", "norm_Y"]]
            final_map.columns = ["resource_id", "X", "Y"]
            end = time.time()
            progress.advance(task)

        topic_volumes["X"] = topic_volumes["topic_volume"].apply(
            lambda x: (x - x_min) / intervals_x
        )

        map_details = {
            "map_id": self.map_id,
            "time_taken": (end - start),
            "num_resources": len(final_map["resource_id"].unique()),
            "num_topics": len(topics["topic_name"].unique()),
            "num_levels": self.map_config.NUM_LEVELS,
            "X_Axis": {
                "volume_range": {
                    "min_volume": min(topics["topic_volume"].values),
                    "max_volume": max(topics["topic_volume"].values),
                },
                "interval": intervals_x,
                "start": min(final_map["X"].values),
                "end": max(final_map["X"].values),
            },
            "Y_Axis": {
                "volume_range": {
                    "min_volume": min(
                        corpus_with_resource_volumes["resource_volume"].values
                    ),
                    "max_volume": max(
                        corpus_with_resource_volumes["resource_volume"].values
                    ),
                },
                "interval": intervals_y,
                "start": min(final_map["Y"].values),
                "end": max(final_map["Y"].values),
            },
        }
        console.log("All Done. Map Summary:")
        console.log(map_details, justify="center")
        return topic_volumes, c_map, resource_topic_mapping, map_details

    def add_resource(self, topics_df):
        """
        Args:
            topics_df: Topics Identified in the Existing Map

        Returns:
            Returns the below objects
            1. A dataframe consisting of the resource locations on the map.
            2. A dataframe consisting of the mapping between the resources and the topics
            3. A JSON Object containing some properties of the map.

        """
        map_details = {}
        start = time.time()
        timestamp = str(datetime.now().strftime("%Y%m%d_%H-%M-%S"))
        console.log(f"Found {len(self.corpus)} Learning Resources.!!")
        console.log("Cleaning Corpus[1/7]..")
        preprocessed_texts_path = Path.joinpath(
            self.map_path, f"new_resources_{timestamp}_preprocessed_texts.obj"
        )
        clean_corpus = self.get_clean_corpus(self.corpus)
        pd.to_pickle(clean_corpus, preprocessed_texts_path)
        console.log(clean_corpus)
        console.log("Load Topic Model for the map[2/7]...")
        topic_model = self.get_topic_model(clean_corpus["tokens"], create=False)

        with Progress() as progress:
            task = progress.add_task(
                total=7, description="Adding Resources to Competency Map"
            )
            progress.advance(task)
            progress.advance(task)
            console.log(
                "Get Resource Embedding Model from Cleaned Corpus[3/7]....."
            )
            document_embedding_model = self.get_document_embedding_model(
                clean_corpus["tokens"], create=False
            )
            progress.advance(task)
            console.log("Get Resource Volume for each resource[4/7].....")
            corpus_with_resource_volumes = self.get_resource_volumes(
                clean_corpus, document_embedding_model
            )
            console.log(f"Columns: ${corpus_with_resource_volumes.columns}")
            progress.advance(task)
            # Map Resources to Topics
            console.log("Map Each Resource to set of topics[5/7]......")
            resource_topic_mapping = self.get_resource_mapping(
                corpus_with_resource_volumes, topic_model, topics_df
            )
            console.log(f"Columns: ${resource_topic_mapping.columns}")
            progress.advance(task)
            # Obtain Best X-Mapping for X
            console.log("Get Best Mapping for each resource[6/7].......")
            resource_mapping_with_best_topic = self.map_resource_to_best_topic(
                resource_topic_mapping
            )
            console.log(f"Columns: ${resource_mapping_with_best_topic.columns}")
            c_map = resource_mapping_with_best_topic[
                ["resource_id", "X", "Y"]
            ].drop_duplicates()
            progress.advance(task)
            # Scale Location to Competency Map Space
            console.log("Build Competency Map[7/7]........")
            x_min, intervals_x = self.get_intervals(
                topics_df["topic_volume"],
                len(topics_df["topic_name"].unique()),
            )
            c_map["norm_X"] = c_map["X"].apply(
                lambda x: (x - x_min) / intervals_x
            )
            y_min, intervals_y = self.get_intervals(
                c_map["Y"], self.map_config.NUM_LEVELS
            )
            c_map["norm_Y"] = c_map["Y"].apply(
                lambda y: (y - y_min) / intervals_y
            )
            final_map = c_map[["resource_id", "norm_X", "norm_Y"]]
            final_map.columns = ["resource_id", "X", "Y"]
            end = time.time()
            progress.advance(task)

        map_details = {
            "map_id": self.map_id,
            "time_taken": (end - start),
            "num_resources": len(final_map["resource_id"].unique()),
        }
        console.log("All Done. Map Details")
        console.log(map_details, justify="center")
        return c_map, resource_topic_mapping, map_details

    def refresh_map(self):
        """Recreate the Map by clearing all the trained models"""
        os.removedirs(self.model_directory)
        self.create_map()

    def delete_resource(self, resource_ids):
        """
        Deletes a resource from a existing competency map
        Args:
            resource_ids: Resource IDs to be deleted.
        Returns:

        """
        # TODO: Deleting Resources to a existing competency map.
        pass
