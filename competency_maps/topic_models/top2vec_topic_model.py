from pathlib import Path

import pandas as pd
import pyLDAvis.gensim
from rich import print
from rich.panel import Panel
from rich.progress import track
from top2vec import Top2Vec

from competency_maps.topic_models.topic_model import TopicModel


class Top2vecTopicModel(TopicModel):
    def __init__(
        self,
        text_corpus,
        min_topic_clusters,
        max_topic_clusters,
        enable_tfidf,
        enable_filter,
    ):
        """Initialise the LDA Topic Model"""
        TopicModel.__init__(
            self,
            text_corpus,
            min_topic_clusters,
            max_topic_clusters,
            enable_tfidf,
            enable_filter,
        )
        self.topic_model_type = "top2vec"

    def evaluate_models(self):
        """Create the LDA Models ranging from 0 to max_topic_clusters
        and identify the best model based on the evaluation metric"""
        self.console.log(f"[bold_blue]Corpus Length: {len(self.corpus)}[/]")
        text_corpus = [" ".join(word) for word in self.doc_tokens]
        top2vec_model = Top2Vec(text_corpus, speed="fast-learn")
        self.console.log(
            f"Model with {top2vec_model.get_num_topics()} Clusters found..."
        )
        self.best_model = top2vec_model
        self.best_model_clusters = top2vec_model.get_num_topics()
        self.model_list[top2vec_model.get_num_topics()] = top2vec_model
        return self.model_list

    def train_model(self):
        """Train the LDA Model"""
        topic_model = Top2Vec(self.corpus, speed="deep-learn")
        return topic_model

    def load_model(self, model_path, dictionary_path):
        """Load a LDA Model given the model path and dictionary path"""
        self.best_model = Top2Vec.load(model_path)
        self.best_model_clusters = self.best_model.get_num_topics()

    def topic_mappings(self):
        """Return the topic cluster that the documents in the corpus are mapped to."""
        doc_top, doc_dist = self.best_model._calculate_documents_topic(
            self.best_model.topic_vectors,
            self.best_model._get_document_vectors(),
            dist=True,
        )
        doc_scores = []
        for idx in range(0, len(doc_top)):
            doc_scores.append([(doc_top[idx], doc_dist[idx])])
        return doc_scores

    def get_best_model(self, output_dir=None, metric="cv_score"):
        self.evaluate_models()
        self.best_model.save(
            Path.joinpath(
                output_dir, f"best_{self.topic_model_type}_topic_model.obj",
            ).as_posix()
        )
        return self.best_model

    def get_topics(self, topic_list, relevance_value=0.5, num_keywords=30):
        topics_list = []
        topic_sizes, topic_nums = self.best_model.get_topic_sizes()
        for topic_num in topic_nums:
            word_score_dict = dict(
                zip(
                    self.best_model.topic_words[topic_num],
                    self.best_model.topic_word_scores[topic_num],
                )
            )
            for key, val in word_score_dict.items():
                topics_list.append([topic_num, key, val])
        topics_df = pd.DataFrame(
            topics_list,
            columns=[
                "topic_cluster_id",
                "topic_name",
                "topic_cluster_probability",
            ],
        )
        topics_df["topic_type"] = "Relevant"
        if num_keywords == -1:
            return topics_df
        else:
            final_df = (
                topics_df.sort_values(
                    "topic_cluster_probability", ascending=False
                )
                .groupby("topic_cluster_id")
                .head(num_keywords)
            )
            return final_df
