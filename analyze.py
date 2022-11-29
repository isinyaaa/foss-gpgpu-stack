#!/usr/bin/env python3

import logging
import pandas as pd
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
LOGGER = logging.getLogger()


class Preprocess:
    def run(self, data):
        data.contents = data.contents.apply(self.clean)
        return self.vectorize(data)

    def clean(self, text):
        import re

        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'[%s]' % re.escape("""<>'"()[]{}"""), ' ', text)
        text = re.sub(r'\d*', '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'(?:(?:https?:\/\/)|(?:www\.))([^\/ ]+)(?:(?:\.[a-zA-Z]*)|(?::\d{3,5}))(\/+\S*)?', r'\1 \2', text)
        text = re.sub(r'[%s]' % re.escape("""!"'#$%&*+.,/:;=?@\-^`|~"""), ' ', text)
        text = re.sub(r'\.+ ', ' ', text)
        text = re.sub(r' \S ', ' ', text)

        return text

    def vectorize(self, data):
        import nltk
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')
        from nltk import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        from sklearn.feature_extraction.text import CountVectorizer

        stop_words = stopwords.words('english')
        stop_words.append('www')

        class LemmaTokenizer:
            def __init__(self):
                self.wnl = WordNetLemmatizer()

            def __call__(self, doc):
                return [ self.wnl.lemmatize(t)
                            for t in word_tokenize(doc)
                                if t not in stop_words ]

        vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                     ngram_range=(1, 3), min_df=3, max_df=0.8,
                                     lowercase=True, strip_accents='ascii',
                                     analyzer='word', max_features=1000)

        return vectorizer, vectorizer.fit_transform(data.contents.values)


class Cloud:
    """
    Generate a word cloud from the given data
    """
    def run(self, vectorizer, data):
        frequencies = self.prepare(vectorizer, data)
        wordcloud = self.gencloud(frequencies)
        self.show(wordcloud)

    def prepare(self, vectorizer, data):
        import numpy as np

        frequencies = dict(
                zip(np.array(vectorizer.get_feature_names_out()),
                    data.toarray().sum(axis=0))
                )

        return frequencies

    def gencloud(self, frequencies):
        """
        Generate the word cloud
        """
        from wordcloud import WordCloud

        logging.info('Generating word cloud')

        return WordCloud(background_color="white", max_words=5000, width=1600,
                         height=800, contour_width=3,
                         contour_color='steelblue').fit_words(frequencies)

    def show(self, wordcloud):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.figure(figsize = (2, 1), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)

        plt.show()


class LDA:
    def save_result_as_html(self, prepare, *args, **kwargs):
        import pyLDAvis
        import pickle

        # pyLDAvis.enable_notebook()
        LDAvis_filepath = os.path.join('./ldavis_prepared_' + str(self.topics))

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

        super().save_result_as_html(prepare, model, data, vectorizer, mds='tsne')


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

        super().save_result_as_html(prepare, model, corpus, id2word, None)

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


def main(args):
    readme_data = pd.read_csv(args.input_file)
    logging.debug(readme_data.head())

    vectorizer, processed_data = Preprocess().run(readme_data)

    if args.cloud:
        Cloud().run(vectorizer, processed_data)

    if args.hyperparams:
        lda = GensimLDA(args.input_file, readme_data, args.alpha, args.eta,
                        args.topics, args.workers)

        lda.test_model_hyperparams(vectorizer, processed_data,
                                   args.min_alpha, args.max_alpha, args.alpha_step,
                                   args.min_eta, args.max_eta, args.eta_step,
                                   args.min_topics, args.max_topics, args.topics_step)
    if args.gensim:
        GensimLDA(args.input_file, readme_data,
                  args.alpha, args.eta, args.topics,
                  args.workers).run(vectorizer, processed_data)
    else:
        SKLearnLDA(args.topics,
                   args.workers).run(vectorizer, processed_data)



if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from multiprocessing import cpu_count

    parser = ArgumentParser(prog='lda.py',
                            description='Latent Dirichlet Analysis for Repositories',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action='store_true',
                       help='increase output verbosity')
    group.add_argument('-d', '--debug', action='store_true',
                       help='enable debug logging')
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(),
                        help='number of workers to use')
    parser.add_argument('-c', '--cloud', action='store_true',
                        help='generate word cloud')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_lda = subparsers.add_parser('lda', help='run LDA')
    lda_group = parser_lda.add_mutually_exclusive_group(required=True)
    lda_group.add_argument('-gs', '--gensim', action='store_true',
                           help='use Gensim implementation')
    lda_group.add_argument('-sk', '--sklearn', action='store_true',
                           help='use sklearn implementation')
    parser_lda.add_argument('-t', '--topics', type=int, default=10,
                            help='number of topics to use')
    parser_lda.add_argument('-a', '--alpha', type=float, default=0.8,
                            help='alpha to use')
    parser_lda.add_argument('-e', '--eta', type=float, default=0.8,
                            help='eta to use')
    parser_lda.add_argument('-tw', '--top-words', type=float, default=20,
                            help='top words to get on NMF')
    parser_lda.set_defaults(hyperparams=False) # HACK

    parser_hy = subparsers.add_parser('hy', help='run hyperparameter tuning on Gensim LDA')
    parser_hy.add_argument('-mt', '--min-topics', type=int, default=2,
                           help='minimum number of topics to test')
    parser_hy.add_argument('-xt', '--max-topics', type=int, default=11,
                           help='maximum number of topics to test')
    parser_hy.add_argument('-ts', '--topics-step', type=int, default=1,
                           help='step size for topic testing')
    parser_hy.add_argument('-ma', '--min-alpha', type=float, default=0.1,
                           help='minimum alpha to test')
    parser_hy.add_argument('-xa', '--max-alpha', type=float, default=1,
                           help='maximum alpha to test')
    parser_hy.add_argument('-as', '--alpha-step', type=float, default=0.3,
                           help='step size for alpha testing')
    parser_hy.add_argument('-me', '--min-eta', type=float, default=0.1,
                           help='minimum eta to test')
    parser_hy.add_argument('-xe', '--max-eta', type=float, default=1,
                           help='maximum eta to test')
    parser_hy.add_argument('-es', '--eta-step', type=float, default=0.3,
                           help='step size for eta testing')
    parser_hy.set_defaults(hyperparams=True,
                           alpha=None, eta=None, topics=None) # HACK

    parser.add_argument('input_file', help='input file')

    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.INFO)
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    main(args)
