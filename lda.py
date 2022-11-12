import logging
import pandas as pd
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
LOGGER = logging.getLogger()


class Cloud:
    """
    Generate a word cloud from the given data
    """

    def prepare(self, data):
        """
        Prepare the data for the word cloud
        """
        import re

        return data['contents'].map(lambda x: re.sub(r'[,\.!?]', '', x)).map(lambda x: x.lower())

    def gencloud(self, words):
        """
        Generate the word cloud
        """
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        long_string = ','.join(list(words))

        logging.info('Generating word cloud')

        wordcloud = WordCloud(background_color="white", max_words=5000,
                              contour_width=3, width=1600, height=800,
                              contour_color='steelblue').generate(long_string)

        # plot the WordCloud image
        plt.figure(figsize = (2, 1), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)

        plt.show()


class LDA:
    """
    Generate a LDA model from the given data
    """
    def __init__(self, input_file, alpha=0.5, eta=0.5, topics=10):
        self.input_file = input_file
        self.alpha = alpha
        self.eta = eta
        self.topics = topics

    def prepare(self, data):
        from gensim import corpora
        import spacy

        data = data.contents.values.tolist()

        def remove_stopwords(texts):
            from gensim.utils import simple_preprocess
            import nltk
            nltk.download('stopwords')
            from nltk.corpus import stopwords

            stop_words = stopwords.words('english')
            stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'http',
                               'https', 'com', 'www', 'org', 'net', 'edu',
                               'gov', 'io'])
            return [[word for word in simple_preprocess(str(doc), deacc=True)
                     if word not in stop_words] for doc in texts]

        data = remove_stopwords(data)

        def make_ngrams(texts):
            from gensim.models import Phrases, phrases

            bigram = phrases.Phraser(Phrases(data, min_count=5, threshold=100))
            trigram = phrases.Phraser(Phrases(bigram[data], threshold=100))

            texts = [bigram[doc] for doc in texts]

            return [trigram[bigram[doc]] for doc in texts]

        data = make_ngrams(data)

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        data = lemmatization(data)

        logging.debug("Data entries after LDA prepare")
        logging.debug(data[:1][0][:30])

        id2word = corpora.Dictionary(data)
        corpus = [id2word.doc2bow(text) for text in data]

        logging.debug("Corpus entries after LDA prepare")
        logging.debug(data[:1][0][:30])

        return data, corpus, id2word

    def train(self, corpus, id2word,
              alpha=None, eta=None, topics=None):
        from gensim.models import LdaMulticore

        if alpha is None:
            alpha = self.alpha
        if eta is None:
            eta = self.eta
        if topics is None:
            topics = self.topics

        lda_model = LdaMulticore(corpus=corpus, id2word=id2word,
                                 num_topics=topics, random_state=100,
                                 chunksize=100, passes=10, alpha=alpha,
                                 eta=eta, per_word_topics=True, workers=15)

        if LOGGER.getEffectiveLevel() == logging.DEBUG:
            from pprint import pprint
            pprint(lda_model.print_topics())

        return lda_model

    def test_model_hyperparams(self, data, corpus, id2word):
        from gensim.utils import ClippedCorpus
        import numpy as np
        import tqdm
        from datetime import datetime

        corpus_sets = [ClippedCorpus(corpus, int(len(corpus) * 0.75)), corpus]
        corpus_title = ['75% Corpus', '100% Corpus']

        topics_range = range(2, 11)

        alpha = list(np.arange(0.1, 1.0, 0.3))
        alpha.append('symmetric')
        alpha.append('asymmetric')

        eta = list(np.arange(0.1, 1.0, 0.3))
        eta.append('symmetric')

        model_results = {
            'Validation_Set': [],
            'Topics': [],
            'Alpha': [],
            'Eta': [],
            'Coherence': []
        }

        def model_perplexity(data, id2word, lda_model):
            """
            Get model perplexity for analysis
            """
            from gensim.models import CoherenceModel

            return CoherenceModel(model=lda_model, texts=data, dictionary=id2word,
                                  coherence='c_v').get_coherence()

        iterations = len(eta) * len(alpha) * len(topics_range) * len(corpus_title)
        pbar = tqdm.tqdm(total=iterations, desc="Running LDA")

        for i in range(len(corpus_sets)):
            for k in topics_range:
                for a in alpha:
                    for b in eta:
                        logging.info("Running LDA with k=%d, a=%s, b=%s" % (k, a, b))

                        lda_model = self.train(corpus_sets[i], id2word,
                                               a, b, k)
                        cv = model_perplexity(data, id2word, lda_model)

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

    def visualize(self, corpus, id2word, lda_model):
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
        import pickle

        # pyLDAvis.enable_notebook()
        LDAvis_filepath = os.path.join('./ldavis_prepared_' + str(self.topics))

        if True:
            LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
            with open(LDAvis_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)

        with open(LDAvis_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)

        pyLDAvis.save_html(LDAvis_prepared,
                           os.path.join(LDAvis_filepath) + '.html')

        # pyLDAvis.show(LDAvis_prepared)

def main(args):
    readme_data = pd.read_csv(args.input_file)
    logging.debug(readme_data.head())
    readme_data = readme_data.drop(columns=['filename'], axis=1)
    logging.debug(readme_data.head())

    if args.cloud:
        cloud = Cloud()
        words = cloud.prepare(readme_data)
        cloud.gencloud(words)

    lda = LDA(args.input_file, 0.8, 0.8, 10)
    data, corpus, id2word = lda.prepare(readme_data)

    if args.hyperparams:
        lda.test_model_hyperparams(data, corpus, id2word)
    else:
        lda_model = lda.train(corpus, id2word)
        lda.visualize(corpus, id2word, lda_model)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(prog='lda.py',
                            description='Latent Dirichlet Analysis for Repositories',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase output verbosity')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='enable debug logging')
    parser.add_argument('-hy', '--hyperparams', action='store_true', default=False,
                        help='test hyperparams')
    parser.add_argument('-c', '--cloud', action='store_true', default=False,
                        help='generate word cloud')

    parser.add_argument('input_file', help='input file')

    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.INFO)
    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    main(args)
