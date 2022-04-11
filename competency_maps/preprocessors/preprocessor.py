import re
import unicodedata
from abc import ABCMeta, abstractmethod

from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import track

from competency_maps.preprocessors import contractions


class Preprocessor(metaclass=ABCMeta):
    """ Perform Standard Text Preprocessing Techniques on the learning resource content

    Attributes:
        html_stripping (bool): Strip HTML Tags from the text
        contraction_expansion(bool): Expand Contractions in the text
        accented_char_removal(bool): Remove Non-UTF8 characters from the text
        text_lower_case (bool): Normalise the case by converting all text to lower
        text_lemmatization(bool): Perform Lemmatization on the text to obtain the lemmas for each token
        special_char_removal(bool): Remove Special Characters from the text
        stopword_removal(bool): Remove the stopwords from the text
        remove_digits(bool): Remove digits from the text
    """

    def __init__(
        self,
        html_stripping=True,
        contraction_expansion=True,
        accented_char_removal=True,
        text_lower_case=True,
        text_lemmatization=True,
        special_char_removal=True,
        stopword_removal=True,
        remove_digits=True,
    ):
        self.html_stripping = html_stripping
        self.contraction_expansion = contraction_expansion
        self.accented_char_removal = accented_char_removal
        self.text_lower_case = text_lower_case
        self.text_lemmatization = text_lemmatization
        self.special_char_removal = special_char_removal
        self.stopword_removal = stopword_removal
        self.remove_digits = remove_digits
        self.console = Console()

    @abstractmethod
    def lemmatize_text(self, doc):
        pass

    @abstractmethod
    def remove_stopwords(self, doc):
        pass

    @abstractmethod
    def tokenize(self, doc):
        pass

    @staticmethod
    def strip_html_tags(text):
        """Strips the HTML Tags from the text"""
        try:
            soup = BeautifulSoup(text, "html.parser")
            stripped_text = soup.get_text()
        except:
            stripped_text = text
        return stripped_text

    @staticmethod
    def remove_accented_chars(text):
        """Remove Non UTF-8 Characters from text"""
        try:
            text = (
                unicodedata.normalize("NFKD", text)
                .encode("ascii", "ignore")
                .decode("utf-8", "ignore")
            )
            return text
        except:
            return text

    @staticmethod
    def expand_contractions(
        text, contraction_mapping=contractions.CONTRACTION_MAP
    ):
        """Replace contractions in string of text"""
        contractions_pattern = re.compile(
            "({})".format("|".join(contraction_mapping.keys())),
            flags=re.IGNORECASE | re.DOTALL,
        )

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = (
                contraction_mapping.get(match)
                if contraction_mapping.get(match)
                else contraction_mapping.get(match.lower())
            )
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        try:
            expanded_text = contractions_pattern.sub(expand_match, text)
            expanded_text = re.sub("'", "", expanded_text)
        except:
            expanded_text = text
        return expanded_text

    @staticmethod
    def remove_special_characters(text, remove_digits):
        """Remove Special Characters from text

        Args:
            remove_digits: Boolean indicating whether to remove digits or not.
        """
        pattern = r"[^a-zA-z0-9\s]" if not remove_digits else r"[^a-zA-z\s]"
        text = re.sub(pattern, "", text)
        return text

    def normalize_corpus(self, corpus):
        """ Apply all preprocessing steps as defined by the attributes on the corpus"""
        normalized_corpus = []
        counter = 0
        for idx in track(
            range(len(corpus)), description="Preprocessing Documents..."
        ):
            doc = corpus[idx]
            # print("Index: {0}".format(counter))
            counter += 1
            # print("Content: {0}".format(doc))
            # strip HTML
            if self.html_stripping:
                doc = self.strip_html_tags(doc)
            # remove accented characters
            if self.accented_char_removal:
                doc = self.remove_accented_chars(doc)
            # expand contractions
            if self.contraction_expansion:
                doc = self.expand_contractions(doc)
            # lowercase the text
            if self.text_lower_case:
                doc = doc.lower()
            # remove extra newlines
            doc = re.sub(r"[\r|\n|\r\n]+", " ", doc)
            # lemmatize text
            if self.text_lemmatization:
                doc = self.lemmatize_text(doc)
                # doc = self.get_noun_lemmas(doc)
            # remove special characters and\or digits
            if self.special_char_removal:
                # insert spaces between special characters to isolate them
                special_char_pattern = re.compile(r"([{.(-)!}])")
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = self.remove_special_characters(
                    doc, remove_digits=self.remove_digits
                )
            # remove extra whitespace
            doc = re.sub(" +", " ", doc)
            # remove stopwords
            if self.stopword_removal:
                doc = self.remove_stopwords(doc)
            normalized_corpus.append(doc)
        return normalized_corpus

    def get_tokens(self, corpus):
        """Obtain Tokens for each document in the corpus"""
        tokenized_doc = []
        for doc in corpus:
            tokens = self.tokenize(doc)
            filtered_tokens = [w for w in tokens if len(w) > 2]
            tokenized_doc.append(filtered_tokens)
        return tokenized_doc
