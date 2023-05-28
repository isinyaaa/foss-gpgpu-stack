import logging


class Preprocess:
    def run(self, data):
        data.readmes_contents = data.readmes_contents.apply(self.clean)
        logging.info("Cleaning done")
        logging.debug("--------------")
        logging.debug(data.head())
        logging.info("Vectorizing...")
        return self.vectorize(data)

    def clean(self, text):
        import re

        text = re.sub(r"[%s]" % re.escape(""""'"()[]{}<>"""), " ", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\n", "", text)
        text = re.sub(r"(?:(?:https?:\/\/)|(?:www\.))([^\/ ]+)(?:(?:\.[a-zA-Z]*)|(?::\d{3,5}))(\/+\S*)?", r"\1 \2", text)
        text = re.sub(r"[%s]" % re.escape("""!#$%&*+.,/:;=?@\_^`|~"""), " ", text)
        text = re.sub(r"\s+", " ", text)

        return text

    def vectorize(self, data):
        import nltk
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("omw-1.4")
        nltk.download("stopwords")
        from nltk import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from sklearn.feature_extraction.text import CountVectorizer

        stop_words = stopwords.words("english")
        stop_words.extend(["www", "github", "file", "install", "doc",
                           "documentation", "option", "also", "following",
                           "release", "project", "package", "example",
                           "build", "run", "build"])

        class LemmaTokenizer:
            def __init__(self):
                self.wnl = WordNetLemmatizer()

            def __call__(self, doc):
                return [
                        self.wnl.lemmatize(t)
                        for t in word_tokenize(doc)
                        if t not in stop_words
                        ]

        vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                     ngram_range=(1, 3), min_df=0.1, max_df=0.8,
                                     lowercase=True, strip_accents="ascii",
                                     analyzer="word", max_features=1000)

        return vectorizer, vectorizer.fit_transform(data.readmes_contents.values)
