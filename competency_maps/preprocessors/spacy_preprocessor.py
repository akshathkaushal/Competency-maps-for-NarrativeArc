import spacy

from competency_maps.preprocessors import Preprocessor


class SpacyPreprocessor(Preprocessor):
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
        """ Initiliase the preprocessing steps to be performed

        Additionally, download the language model for the english language if not already present.

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
        if spacy.info().get("Models") == "en":
            print("Found Model")
        else:
            spacy.cli.download("en")
        self.nlp = spacy.load("en_core_web_sm")

    def lemmatize_text(self, text):
        """Lemmatize the text and also exclude Words with that are pronouns"""
        text = self.nlp(text)
        text = " ".join(
            [
                word.lemma_ if word.lemma_ != "-PRON-" else word.text
                for word in text
            ]
        )
        return text

    def get_noun_lemmas(self, text):
        """Lemmatize the text and retain only Noun Phrases from the text"""
        doc = self.nlp(text)
        tokens = [token for token in doc]
        noun_tokens = [
            token
            for token in tokens
            if token.tag_ == "NN" or token.tag_ == "NNP" or token.tag_ == "NNS"
        ]
        noun_lemmas = [
            noun_token.lemma_
            for noun_token in noun_tokens
            if noun_token.is_alpha
        ]
        return " ".join(noun_lemmas)

    def remove_stopwords(self, text):
        """Remove Stop words from the text"""
        text = self.nlp(text)
        text = " ".join([word.text for word in text if not word.is_stop])
        return text

    def tokenize(self, text):
        """Obtain tokens from the text"""
        tokens = self.nlp(text)
        id_sequence = map(lambda x: x.orth, [token for token in tokens])
        token_list = map(
            lambda x: self.nlp.vocab[x].text, [id for id in id_sequence]
        )
        return token_list
