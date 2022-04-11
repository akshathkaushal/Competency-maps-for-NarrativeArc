import os

import nltk

from competency_maps.preprocessors import Preprocessor


class NltkPreprocessor(Preprocessor):
    """ Use NLTK Library to perform the preprocessing pipelines."""

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
        """Initialise the preprocessing steps to be performed.

        Additionally, also download the other components that are required for preprocessing if they dont already exist:
        1. corpora
        2. stopwords
        3. tokenizers
        4. punkt

        We use the ToktokTokenizer for obtaining the tokens, WordNetLemmatizer to obtain the lemmas and
        the PorterStemmer for stemming the text.
        """
        Preprocessor.__init__(
            self,
            html_stripping,
            contraction_expansion,
            accented_char_removal,
            text_lower_case,
            text_lemmatization,
            special_char_removal,
            stopword_removal,
            remove_digits,
        )
        extensions = [("corpora", "stopwords"), ("tokenizers", "punkt")]
        paths = (
            nltk.data.path
        )  # there are usually quite a few, so we check them all.
        missing = []
        for ext in extensions:
            ext_found = False
            print("Looking for " + ext[1])
            for path in paths:
                if os.path.exists(os.path.join(path, ext[0], ext[1])):
                    ext_found = True
                    print("Found " + ext[1])
                    break
            if not ext_found:
                print("Missing " + ext[1])
                missing.append(ext)
        for ext_tuple in missing:
            nltk.download(ext_tuple[1])
        self.stopword_list = nltk.corpus.stopwords.words("english")
        self.stopword_list.remove("no")
        self.stopword_list.remove("not")
        self.tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.stemmer = nltk.porter.PorterStemmer()

    def simple_stemmer(self, text):
        """Perform stemming on the text using the Porter Stemmer"""
        text = " ".join([self.stemmer.stem(word) for word in text.split()])
        return text

    def lemmatize_text(self, text):
        """Lemmatize the text using the WordNetLemmatizer"""
        tokens = self.tokenizer.tokenize(text)
        text = " ".join([self.lemmatizer.lemmatize(word) for word in tokens])
        return text

    def remove_stopwords(self, text):
        """Remove Stopwords defined in the stopwords corpus"""
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [
            token for token in tokens if token not in self.stopword_list
        ]
        filtered_text = " ".join(filtered_tokens)
        return filtered_text

    def tokenize(self, text):
        """Use the ToktokTokenizer to obtain tokens from the text"""
        tokens = self.tokenizer.tokenize(text)
        return tokens
