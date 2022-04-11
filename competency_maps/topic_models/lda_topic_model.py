import copy

import gensim
import pandas as pd
import pyLDAvis.gensim
from rich import print
from rich.panel import Panel
from rich.progress import track

from competency_maps.topic_models.topic_model import TopicModel


class LdaTopicModel(TopicModel):
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
        self.topic_model_type = "lda"

    def evaluate_models(self):
        """Create the LDA Models ranging from 0 to max_topic_clusters
        and identify the best model based on the evaluation metric"""
        self.console.log(f"[bold_blue]Num Topics: {self.max_topic_clusters}[/]")
        self.console.log(f"[bold_blue]Corpus Length: {len(self.corpus)}[/]")
        cv_scores = []
        umass_scores = []
        lp_scores = []
        for topics in track(
            range(self.min_topic_clusters, self.max_topic_clusters + 1),
            description="Training Topic Model...",
        ):
            self.console.log(f"[yellow]Running LDA with {topics} topic(s)[/]")
            lm = gensim.models.LdaMulticore(
                corpus=self.corpus,
                num_topics=topics,
                id2word=self.dictionary,
                passes=100,
                chunksize=2000,
                alpha="asymmetric",
                eval_every=10,
                random_state=42,
            )
            self.model_list[topics] = lm
            cm_1 = gensim.models.CoherenceModel(
                model=lm,
                texts=self.doc_tokens,
                dictionary=self.dictionary,
                coherence="c_v",
                processes=1,
            ).get_coherence()
            cm_2 = gensim.models.CoherenceModel(
                model=lm,
                corpus=self.corpus,
                dictionary=self.dictionary,
                coherence="u_mass",
                processes=1,
            ).get_coherence()
            lp = lm.log_perplexity(self.corpus)
            lp_scores.append(lp)
            print(
                Panel(
                    f"[red]CV Score: {cm_1}[/]\n"
                    f"[green]u_mass score: {cm_2}[/]\n"
                    f"[blue]log perplexity: {lp}[/]\n"
                )
            )
            cv_scores.append(cm_1)
            umass_scores.append(cm_2)
        self.metrics = pd.DataFrame(
            {
                "topics": range(
                    self.min_topic_clusters, self.max_topic_clusters + 1
                ),
                "cv_score": cv_scores,
                "umass_score": umass_scores,
                "log_perplexity": lp_scores,
            }
        )
        return self.model_list

    def train_model(self):
        """Train the LDA Model"""
        lm = gensim.models.LdaModel(
            corpus=self.corpus,
            num_topics=self.max_topic_clusters,
            id2word=self.dictionary,
            passes=100,
            random_state=42,
        )
        return lm

    def load_model(self, model_path, dictionary_path):
        """Load a LDA Model given the model path and dictionary path"""
        self.console.log(f"[yellow]Loading Existing Topic Model[/]")
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
