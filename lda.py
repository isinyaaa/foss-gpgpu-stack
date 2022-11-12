import pandas as pd

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

        print("Data entries after LDA prepare")
        print(data[:1][0][:30])

        id2word = corpora.Dictionary(data)
        corpus = [id2word.doc2bow(text) for text in data]

        print("Corpus entries after LDA prepare")
        print(data[:1][0][:30])

        return corpus, id2word

    def train(self, corpus, id2word):
        from gensim.models.ldamodel import LdaModel

        lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=id2word,
                             alpha='auto', eta='auto', iterations=100,
                             passes=10, per_word_topics=True)

        return lda_model

    def visualize(self, corpus, id2word, lda_model):
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
        import pickle
        import os

        # pyLDAvis.enable_notebook()
        LDAvis_filepath = os.path.join('./ldavis_prepared_' + str(10))

        if True:
            LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
            with open(LDAvis_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)

        with open(LDAvis_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)

        pyLDAvis.save_html(LDAvis_prepared,
                           os.path.join(LDAvis_filepath) + '.html')

        # pyLDAvis.show(LDAvis_prepared)

def main():
    readme_data = pd.read_csv('repos/repo_readmes.csv')
    print(readme_data.head())
    readme_data = readme_data.drop(columns=['filename'], axis=1)
    print(readme_data.head())

    cloud = Cloud()
    words = cloud.prepare(readme_data)
    cloud.gencloud(words)

    lda = LDA()
    data, corpus, id2word = lda.prepare(readme_data)

    lda_model = lda.train(corpus, id2word)
    lda.visualize(corpus, id2word, lda_model)


if __name__ == '__main__':
    main()
