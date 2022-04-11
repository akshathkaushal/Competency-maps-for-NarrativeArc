import copy
from abc import ABCMeta, abstractmethod
from pathlib import Path

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis.gensim
from rich.console import Console


class TopicModel(metaclass=ABCMeta):
    """ Build a Topic Model provided the input corpus

    Attributes:
        text_corpus (bool): Corpus to be used for training the toic model
        min_topic_clusters (int): Minimum number of topic clusters to be used while training
        max_topic_clusters (int): Maximum number of topic clusters to be used while training
        enable_tfidf (bool): Indicates whether to apply TF-IDF Filtering for the corpus
        enable_filter (bool): Indicates if the dictionary should be filtered for extreme words
    """

    def __init__(
        self,
        text_corpus,
        min_topic_clusters,
        max_topic_clusters,
        enable_tfidf,
        enable_filter,
    ):
        self.console = Console()
        self.topic_model_type = None
        self.corpus = None
        self.min_topic_clusters = min_topic_clusters
        self.max_topic_clusters = max_topic_clusters
        self.doc_tokens = text_corpus.tolist()
        self.model_list = {}
        self.metrics = pd.DataFrame()
        self.best_model = None
        self.best_model_clusters = None
        self.enable_filter = enable_filter
        self.enable_tfidf = enable_tfidf

    @abstractmethod
    def load_model(self, model_path, dictionary_path):
        """Load the topic model given the model path and dictionary file"""
        pass

    @abstractmethod
    def train_model(self):
        """Train a Topic Model"""
        pass

    @abstractmethod
    def evaluate_models(self):
        """Evaluate multiple topic models and choose the best one"""
        pass

    @abstractmethod
    def topic_mappings(self):
        """Return the topic cluster that the documents in the corpus are mapped to."""
        pass

    def get_best_model(self, output_dir=None, metric="cv_score"):
        """Identify the best topic model and prepare the results to be consumed"""
        self.dictionary = gensim.corpora.Dictionary(self.doc_tokens)
        self.console.log(f"Dictionary Length: {len(self.dictionary)}")
        self.corpus = [
            self.dictionary.doc2bow(text) for text in self.doc_tokens
        ]
        self.console.log(f"enable_filter: {self.enable_filter}")
        self.console.log(f"enable tfidf: {self.enable_tfidf}")
        if self.enable_filter:
            filtered_dict = copy.deepcopy(self.dictionary)
            filtered_dict.filter_extremes(no_below=30, no_above=0.5)
            if len(filtered_dict) > 0:
                self.console.log(f" Filtered Dict: {len(filtered_dict)}")
                self.dictionary = filtered_dict
                self.corpus = [
                    self.dictionary.doc2bow(text) for text in self.doc_tokens
                ]
        self.console.log(f"Dictionary Length: {len(self.dictionary)}")
        if self.enable_tfidf:
            tfidf = gensim.models.TfidfModel(
                self.corpus, dictionary=self.dictionary
            )
            self.corpus = tfidf[self.corpus]
        self.console.log(f"Dictionary Length: {len(self.dictionary)}")
        self.evaluate_models()
        try:
            # find_best_model = KneeLocator(self.metrics['topics'], self.metrics[metric], curve='concave',
            #                               direction='increasing', S=1.0)
            # n_topics = find_best_model.knee
            n_topics = self.metrics[
                self.metrics[metric] == self.metrics[metric].max()
            ]["topics"].values[0]
        except:
            n_topics = self.metrics[
                self.metrics[metric] == self.metrics[metric].max()
            ]["topics"].values[0]

        print(f"Best Model is having {n_topics} topics")
        self.best_model = self.model_list.get(n_topics)
        self.best_model_clusters = n_topics
        if output_dir is not None:
            self.metrics.to_csv(
                Path.joinpath(
                    Path(output_dir),
                    f"{self.topic_model_type}_evaluation_metrics.csv",
                )
            )
            self.best_model.save(
                Path.joinpath(
                    output_dir, f"best_{self.topic_model_type}_topic_model.obj",
                ).as_posix()
            )
            self.dictionary.save(
                Path.joinpath(
                    output_dir,
                    f"best_{self.topic_model_type}_topic_model_dictionary.obj",
                ).as_posix()
            )
            self.plot_results(metric, n_topics, output_dir)

        if len(self.best_model.get_topics()) > 1:
            self.lda_results = pyLDAvis.gensim.prepare(
                self.best_model,
                self.corpus,
                self.dictionary,
                R=len(self.dictionary),
            )
            try:
                pyLDAvis.save_html(
                    self.lda_results,
                    Path.joinpath(
                        output_dir,
                        f"best_{self.topic_model_type}_topic_model_vis.html",
                    ).as_posix(),
                )
            except:
                self.console.log("[red]Cannot Save Results.[/red]")
        else:
            self.console.log(
                "[yellow]Best Model has 1 topic. no Visualization possible[/]"
            )

        return self.best_model

    def plot_results(self, metric, n_topics, output_dir):
        """Write the PyLDAVis plot to a file."""
        plt.clf()
        plt.xlabel("number of topics t")
        plt.ylabel(f"{metric}")
        plt.plot(self.metrics["topics"], self.metrics[metric], "bx-")
        plt.vlines(n_topics, plt.ylim()[0], plt.ylim()[1], linestyles="dashed")
        plt.savefig(Path.joinpath(output_dir, "topic_cluster_metric.png"))

    def get_topics(self, n_terms):
        """ Obtain List of Topics/Keywords from each topic cluster in the topic model

        Args:
            n_terms: Number of terms to be extracted from each cluster.

        Returns:
            :class:`pandas:pandas.DataFrame`
        """
        if n_terms == -1:
            significant_topic_keywords = self._get_topic_keywords(
                [], 0.6, n_terms
            )
            significant_topic_keywords["topic_type"] = "Relevant"
        else:
            keywords = int(n_terms / 3)
            captured_topics = []
            # Get Relevant Topics
            relevant_topics = self._get_topic_keywords(
                captured_topics, 0.6, keywords
            )
            relevant_topics["topic_type"] = "Relevant"
            self.console.log(
                "Found {} Relevant Topics: \n {}".format(
                    relevant_topics.shape[0],
                    relevant_topics["topic_name"].values.tolist(),
                )
            )
            captured_topics = relevant_topics["topic_name"].values.tolist()
            # Get Marker Topics
            marker_topics = self._get_topic_keywords(
                captured_topics, 0, keywords
            )
            marker_topics["topic_type"] = "Marker"
            self.console.log(
                "Found {} Marker Topics: \n {}".format(
                    marker_topics.shape[0],
                    marker_topics["topic_name"].values.tolist(),
                )
            )
            captured_topics += marker_topics["topic_name"].values.tolist()
            # Get Generic Topics
            generic_topics = self._get_topic_keywords(
                captured_topics, 1.0, keywords
            )
            generic_topics["topic_type"] = "Generic"
            self.console.log(
                "Found {} Generic Topics: \n {}".format(
                    generic_topics.shape[0],
                    generic_topics["topic_name"].values.tolist(),
                )
            )
            significant_topic_keywords = pd.concat(
                [relevant_topics, marker_topics, generic_topics]
            )
        self.console.log(f"Found {len(significant_topic_keywords)} Topics...")
        return significant_topic_keywords

    def _get_topic_keywords(
        self, topic_list, relevance_value=0.5, num_keywords=30
    ):
        """Identify the top-n keywords from each cluster"""
        all_topic_info = self.lda_results.topic_info
        topic_info = all_topic_info[
            all_topic_info["Category"] != "Default"
        ].copy()
        topic_info["relevance"] = (
            relevance_value * topic_info["logprob"]
            + (1 - relevance_value) * topic_info["loglift"]
        )
        topic_info["probability"] = np.exp(topic_info["logprob"])
        # Remove Common Words
        topic_info = topic_info[~topic_info.Term.isin(topic_list)]
        # Order based on Relevance and pick top n words
        if num_keywords == -1:
            significant_words = topic_info.sort_values(
                "relevance", ascending=False
            )
        else:
            significant_words = (
                topic_info.sort_values("relevance", ascending=False)
                .groupby("Category")
                .head(num_keywords)
            )
        new_df = significant_words[["Category", "Term", "probability"]]
        new_df["Category"] = new_df["Category"].replace(
            to_replace="Topic", value=r"", regex=True
        )
        new_df.columns = [
            "topic_cluster_id",
            "topic_name",
            "topic_cluster_probability",
        ]
        new_df[["topic_cluster_id"]] = new_df[["topic_cluster_id"]].apply(
            pd.to_numeric
        )
        new_df["topic_cluster_id"] = new_df["topic_cluster_id"] - 1
        # For each topic, retain max cluster probability and drop other mappings
        idx = (
            new_df.groupby(["topic_name"])[
                "topic_cluster_probability"
            ].transform(max)
            == new_df["topic_cluster_probability"]
        )
        final_df = new_df[idx]
        return final_df
