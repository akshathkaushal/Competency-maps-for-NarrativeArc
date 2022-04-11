import copy

import gensim
import pandas as pd
import pyLDAvis.gensim
from rich import print
from rich.panel import Panel

from competency_maps.topic_models.topic_model import TopicModel


class HierarchicalTopicModel(TopicModel):
    def __init__(
        self,
        text_corpus,
        min_topic_clusters,
        max_topic_clusters,
        enable_tfidf,
        enable_filter,
    ):
        """Initialise the Hierarchical Topic Model"""
        TopicModel.__init__(
            self,
            text_corpus,
            min_topic_clusters,
            max_topic_clusters,
            enable_tfidf,
            enable_filter,
        )
        self.topic_model_type = "hierarchical"

    def evaluate_models(self):
        """Create the HDP Model and convert it to the mosr approximate LDA Model"""
        self.console.log("Corpus Length: {}".format(len(self.corpus)))
        self.console.log("Dictionary Size: {}".format(len(self.dictionary)))
        hdp_model = gensim.models.HdpModel(
            self.corpus, id2word=self.dictionary, random_state=42
        )
        lda_model_approx = hdp_model.suggested_lda_model()
        n_topics = len(lda_model_approx.show_topics(formatted=False))
        lda_model = gensim.models.LdaModel(
            corpus=self.corpus,
            num_topics=n_topics,
            id2word=self.dictionary,
            random_state=42,
        )
        self.model_list[n_topics] = lda_model
        # hdp_topics = hdp_model.show_topics(formatted=False)
        # hdp_topics = [[word for word, prob in topic] for topicid, topic in hdp_topics]
        cm_1 = gensim.models.CoherenceModel(
            model=lda_model,
            texts=self.doc_tokens,
            dictionary=self.dictionary,
            coherence="c_v",
        ).get_coherence()
        cm_2 = gensim.models.CoherenceModel(
            model=lda_model,
            corpus=self.corpus,
            dictionary=self.dictionary,
            coherence="u_mass",
        ).get_coherence()
        lp = lda_model.log_perplexity(self.corpus)
        print(
            Panel(
                f"[red]CV Score: {cm_1}[/]\n"
                f"[green]u_mass score: {cm_2}[/]\n"
                f"[blue]log perplexity: {lp}[/]\n"
            )
        )
        self.metrics = pd.DataFrame(
            [[n_topics, cm_1, cm_2, lp]],
            columns=["topics", "cv_score", "umass_score", "log_perplexity"],
        )
        return self.model_list

    def train_model(self):
        """Train a HDP Model using the corpus"""
        self.best_model = gensim.models.HdpModel(
            self.corpus, id2word=self.dictionary, random_state=42
        )

    def load_model(self, model_path, dictionary_path):
        """Load the LDA Model obtained from the path"""
        self.console.log(f"[yellow]Loading Existing Topic Model[/]")
        print(model_path)
        print(dictionary_path)
        self.best_model = gensim.models.LdaModel.load(model_path)
        self.dictionary = gensim.corpora.Dictionary.load(dictionary_path)
        self.best_model_clusters = self.best_model.num_topics

    def topic_mappings(self):
        """Return the topic cluster that the documents in the corpus are mapped to."""
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
        doc_classified = [self.best_model[doc] for doc in self.corpus]
        return doc_classified
