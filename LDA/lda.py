import logging

import pandas as pd
import os

LOGGER = logging.getLogger()

class LDA:
    def save_result_as_html(self, str_append, prepare, *args, **kwargs):
        import pyLDAvis
        import pickle

        # pyLDAvis.enable_notebook()
        LDAvis_filepath = os.path.join(f'./ldavis_prepared_{str_append}_{str(self.topics)}')

        LDAvis_prepared = prepare(*args, **kwargs)
        with open(LDAvis_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

        pyLDAvis.save_html(LDAvis_prepared,
                           os.path.join(LDAvis_filepath) + '.html')

        # pyLDAvis.show(LDAvis_prepared)


class SKLearnLDA(LDA):
    def __init__(self, topics, workers):
        self.topics = topics
        self.workers = workers

    def run(self, vectorizer, data):
        model, results = self.train(data)
        self.save_result_as_html(model, data, vectorizer)
        return results

    def train(self, data):
        from sklearn.decomposition import LatentDirichletAllocation

        if self.workers == None:
            workers = -1
        else:
            workers = self.workers

        lda = LatentDirichletAllocation(n_components=self.topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50., random_state=0,
                                        n_jobs=workers)
        return lda, lda.fit_transform(data)

    def save_result_as_html(self, model, data, vectorizer):
        from pyLDAvis.sklearn import prepare

        super().save_result_as_html('sklearn', prepare, model, data,
                                    vectorizer, mds='tsne')


class GensimLDA(LDA):
    """
    Generate a LDA model from the given data
    """
    def __init__(self, input_file, data, alpha, eta, topics, workers):
        self.input_file = input_file
        self.data = data
        self.alpha = alpha
        self.eta = eta
        self.topics = topics
        self.workers = workers

    def run(self, vectorizer, data):
        _, corpus, id2word = self.prepare(vectorizer, data)
        model = self.train(corpus, id2word)
        self.save_result_as_html(model, corpus, id2word)

    def prepare(self, vectorizer, data):
        from gensim.matutils import Sparse2Corpus
        from gensim.corpora import Dictionary

        corpus = Sparse2Corpus(data, documents_columns=False)

        texts = vectorizer.inverse_transform(data)

        id2word = Dictionary(texts)

        return texts, corpus, id2word

    def train(self, corpus, id2word,
              alpha=None, eta=None, topics=None, workers=None):
        from gensim.models import LdaMulticore

        if alpha is None:
            alpha = self.alpha
        if eta is None:
            eta = self.eta
        if topics is None:
            topics = self.topics
        if workers is None:
            workers = self.workers

        model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=topics,
                             random_state=100, chunksize=100, passes=10,
                             alpha=alpha, eta=eta, per_word_topics=True,
                             workers=workers)

        if LOGGER.getEffectiveLevel() == logging.DEBUG:
            from pprint import pprint
            pprint(model.print_topics())

        return model

    def save_result_as_html(self, model, corpus, id2word):
        from pyLDAvis.gensim_models import prepare

        super().save_result_as_html('gensim', prepare, model, corpus, id2word,
                                    mds='tsne')

    def test_model_hyperparams(self, vectorizer, data,
                               min_alpha=0.1, max_alpha=1, alpha_step=0.3,
                               min_eta=0.1, max_eta=1, eta_step=0.3,
                               min_topics=2, max_topics=11, topics_step=1):
        from gensim.utils import ClippedCorpus
        import numpy as np
        import tqdm
        from datetime import datetime

        data, corpus, id2word = self.prepare(vectorizer, data)

        corpus_sets = [ClippedCorpus(corpus, int(len(corpus) * 0.75)), corpus]
        corpus_title = ['75% Corpus', '100% Corpus']

        topics_range = range(min_topics, max_topics, topics_step)

        alpha = list(np.arange(min_alpha, max_alpha, alpha_step))
        alpha.append('symmetric')
        alpha.append('asymmetric')

        eta = list(np.arange(min_eta, max_eta, eta_step))
        eta.append('symmetric')

        model_results = {
            'Validation_Set': [],
            'Topics': [],
            'Alpha': [],
            'Eta': [],
            'Coherence': []
        }

        def model_perplexity(model, texts, id2word):
            """
            Get model perplexity for analysis
            """
            from gensim.models import CoherenceModel

            return CoherenceModel(model=model, texts=texts, dictionary=id2word,
                                  coherence='c_v').get_coherence()

        iterations = len(eta) * len(alpha) * len(topics_range) * len(corpus_title)
        pbar = tqdm.tqdm(total=iterations, desc="Running LDA")

        for i in range(len(corpus_sets)):
            for k in topics_range:
                for a in alpha:
                    for b in eta:
                        logging.info("Running LDA with k=%d, a=%s, b=%s" % (k, a, b))

                        model = self.train(corpus_sets[i], id2word, a, b, k)
                        cv = model_perplexity(model, data, id2word)

                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Eta'].append(b)
                        model_results['Coherence'].append(cv)

                        pbar.update(1)

        current_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        input_file = self.input_file.split("/")[-1].split(".")[0]
        tuning_results_file = os.path.join('./scores/lda_tuning_results-' + input_file + current_time + '.csv')
        pd.DataFrame(model_results).to_csv(tuning_results_file, index=False)
        pbar.close()
        logging.info("LDA tuning results saved to %s" % tuning_results_file)
