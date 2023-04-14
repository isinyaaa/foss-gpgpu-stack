import logging


class Cloud:
    """Generate a word cloud from the given data."""
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
        """Generate the word cloud."""
        from wordcloud import WordCloud

        logging.info('Generating word cloud')

        return WordCloud(background_color="white", max_words=5000, width=1600,
                         height=800, contour_width=3,
                         contour_color='steelblue').fit_words(frequencies)

    def show(self, wordcloud):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(2, 1), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.show()
