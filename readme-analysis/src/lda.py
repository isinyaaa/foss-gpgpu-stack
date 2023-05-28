from datetime import datetime
import logging
import os

import pandas as pd

LOGGER = logging.getLogger()


class LDA:
    def __init__(self, topics, workers, **kwargs):
        self.topics = topics
        self.workers = workers

    def save_result_as_html(self, str_append, prepare, *args, **kwargs):
        import pickle

        import pyLDAvis

        # pyLDAvis.enable_notebook()
        current_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        LDAvis_filepath = f"./ldavis_data/{str_append}_prepared_{str(self.topics)}_{current_time}"

        LDAvis_prepared = prepare(*args, **kwargs)
        with open(LDAvis_filepath, "wb") as f:
            pickle.dump(LDAvis_prepared, f)

        pyLDAvis.save_html(LDAvis_prepared, LDAvis_filepath + ".html")

        # pyLDAvis.show(LDAvis_prepared)


class SKLearnLDA(LDA):
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    def __init__(self, topics, workers, **kwargs):
        super().__init__(topics, workers, **kwargs)

    def run(self, vectorizer, data):
        model, results = self.train(vectorizer, data)
        self.save_result_as_html(model, data, vectorizer)
        return results

    def train(self, vectorizer, data):
        if self.workers is None:
            workers = -1
        else:
            workers = self.workers

        best_params = self.autotune(data)

        logging.info("Training LDA model...")
        model = self.LDA(n_jobs=workers, **best_params)
        fitted = model.fit(data)

        logging.info("Training done")

        print("Topics in LDA model:")
        feature_names = vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-10 - 1:-1]]))
            print()

        return model, fitted

    def autotune(self, data):
        from numpy import arange
        from sklearn.model_selection import GridSearchCV
        # from cuml.model_selection import GridSearchCV

        if self.workers is None:
            workers = -1
        else:
            workers = self.workers

        param_grid = {
            "n_components": arange(2, 10),
            "learning_decay": arange(0.5, 1, 0.1),
            "learning_method": ["online", "batch"],
            "learning_offset": arange(10, 100, 10),
            "max_iter": [10, 50, 100],
        }

        model = self.LDA(n_jobs=-1)

        logging.info("Autotuning LDA model...")
        grid = GridSearchCV(model, param_grid, cv=5, verbose=3,
                            n_jobs=-1)
        grid.fit(data)
        logging.info("Autotuning done")
        logging.info("Best parameters set found on development set:")
        logging.info(grid.best_params_)

        return grid.best_params_

    def save_result_as_html(self, model, data, vectorizer):
        from pyLDAvis.sklearn import prepare

        super().save_result_as_html("sklearn", prepare, model, data,
                                    vectorizer, mds="tsne")


class GensimLDA(LDA):
    """Generate a LDA model from the given data."""

    from gensim.models import LdaMulticore as LDA

    def __init__(self, topics, workers, **kwargs):
        super().__init__(topics, workers, **kwargs)

        if kwargs.get("input_file") is not None:
            self.input_file = kwargs.get("input_file")
            return

        self.alpha = kwargs.get("alpha")
        self.eta = kwargs.get("eta")

    def run(self, vectorizer, data):
        texts, corpus, id2word = self.prepare(vectorizer, data)
        model = self.train(texts, corpus, id2word)
        self.save_result_as_html(model, corpus, id2word)
        self.print_document_topics(model, corpus)
        self.print_topics(model)

    def prepare(self, vectorizer, data):
        from gensim.corpora import Dictionary
        from gensim.matutils import Sparse2Corpus

        corpus = Sparse2Corpus(data, documents_columns=False)

        texts = vectorizer.inverse_transform(data)

        id2word = Dictionary(texts)

        return texts, corpus, id2word

    def train(self, texts, corpus, id2word):
        from pprint import pprint

        best_params = self.autotune(texts, corpus, id2word)

        logging.info("Training LDA model...")
        model = self.LDA(corpus=corpus, id2word=id2word,
                         random_state=100, chunksize=1000, decay=0.5,
                         iterations=50, passes=10,  # gamma_threshold=0.01,
                         per_word_topics=True,
                         workers=self.workers, **best_params)
        logging.info("LDA model trained")

        # logging.info("LDA model topics:")
        # pprint(model.print_topics())

        return model

    def autotune(self, texts, corpus, id2word):
        from numpy import arange
        from sklearn.model_selection import GridSearchCV
        # from cuml.model_selection import GridSearchCV
        from gensim.sklearn_integration.sklearn_wrapper_gensim_ldamodel import SklearnWrapperLdaModel

        param_grid = {
            "alpha": arange(0.01, 1, 0.1),
            "eta": arange(0.01, 1, 0.1),
            "topics": arange(5, 15),
        }

        def model_perplexity(estimator, X, y=None):
            """Get model perplexity for analysis."""
            from gensim.models import CoherenceModel

            return CoherenceModel(model=estimator, texts=texts, dictionary=estimator.id2word,
                                  coherence="c_v").get_coherence()

        model = SklearnWrapperLdaModel(corpus=corpus, id2word=id2word)

        logging.info("Autotuning LDA model...")
        grid = GridSearchCV(model, param_grid, scoring=model_perplexity, cv=5, verbose=4, n_jobs=-1, error_score='raise')
        grid.fit(corpus)
        logging.info("Autotuning done")
        logging.info("Best parameters set found on development set:")
        logging.info(grid.best_params_)

        return grid.best_params_

    def save_result_as_html(self, model, corpus, id2word):
        from pyLDAvis.gensim_models import prepare

        super().save_result_as_html("gensim", prepare, model, corpus, id2word,
                                    mds="tsne")

    def print_topics(self, model):
        for idx, topic in model.print_topics(-1):
            print("Topic: {} \nWords: {}".format(idx, topic))

    def print_document_topics(self, model, corpus):
        import matplotlib.pyplot as plt

        topics = [model.get_document_topics(doc) for doc in corpus]

        topic_freq = [max(t, key=lambda x: x[1])[0] for t in topics]

        plt.hist(topic_freq,
                 bins=self.topics, density=True, rwidth=0.8)
        plt.ylabel("% of documents")
        plt.xlabel("Topic")
        plt.legend()
        current_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        plt.savefig(f"./ldavis_data/{self.topics}_topics_{current_time}.png", dpi=300)

    def test_model_hyperparams(self, vectorizer, data):
        import numpy as np
        import tqdm
        from gensim.utils import ClippedCorpus

        texts, full_corpus, id2word = self.prepare(vectorizer, data)

        corpus_sets = [
                {
                    "name": "75% Corpus",
                    "data": ClippedCorpus(full_corpus, int(len(full_corpus) * 0.75))
                },
                {
                    "name": "100% Corpus",
                    "data": full_corpus
                }
            ]

        topics_range = range(min_topics, max_topics, topics_step)

        alpha_range = list(np.arange(min_alpha, max_alpha, alpha_step))
        alpha_range.append("symmetric")
        alpha_range.append("asymmetric")

        eta_range = list(np.arange(min_eta, max_eta, eta_step))
        eta_range.append("symmetric")

        model_results = {
            "Validation_Set": [],
            "Topics": [],
            "Alpha": [],
            "Eta": [],
            "Coherence": []
        }

        def model_perplexity(model, corpus):
            """Get model perplexity for analysis."""
            from gensim.models import CoherenceModel

            return CoherenceModel(model=model, texts=texts, dictionary=id2word,
                                  coherence="c_v").get_coherence()

        iterations = len(eta_range) * len(alpha_range) * len(topics_range) * len(corpus_sets)
        pbar = tqdm.tqdm(total=iterations, desc="Running LDA")

        for cset in corpus_sets:
            title, corpus = cset["name"], cset["data"]
            for k in topics_range:
                for a in alpha_range:
                    for b in eta_range:
                        logging.info(f"Running LDA with k={k}, a={a}, b={b}")

                        model = self.train(corpus, id2word, a, b, k)
                        cv = model_perplexity(model, texts)

                        model_results["Validation_Set"].append(title)
                        model_results["Topics"].append(k)
                        model_results["Alpha"].append(a)
                        model_results["Eta"].append(b)
                        model_results["Coherence"].append(cv)

                        pbar.update(1)

        current_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        tuning_results_file = f"./scores/gensim_{current_time}.csv"
        pd.DataFrame(model_results).to_csv(tuning_results_file, index=False)
        pbar.close()
        logging.info("LDA tuning results saved to " + tuning_results_file)
