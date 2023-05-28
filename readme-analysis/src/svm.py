import logging

LOGGER = logging.getLogger()


class SKLearnSVM:
    def __init__(self, workers):
        self.workers = workers

    def run(self, vectorizer, data):
        self.train(vectorizer, data)

    def train(self, vectorizer, data):
        from sklearn.cluster import KMeans
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.svm import SVC

        svclassifier = SVC(kernel='linear', C=1, gamma='scale')
        svclassifier.fit(data)

        kmeans = KMeans(n_clusters=5, random_state=0).fit(data)

        print(kmeans.labels_)
        print(confusion_matrix(kmeans.labels_, svclassifier.predict(data)))
        print(classification_report(kmeans.labels_, svclassifier.predict(data)))

    def autotune(self, vectorizer, data):
        # from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.svm import SVC

        X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        }

        workers = -1 if self.workers is None else self.workers

        grid = GridSearchCV(SVC(), param_grid, cv=5, verbose=LOGGER.level,
                            n_jobs=workers)
        grid.fit(X_train)

        # prediction = grid.best_estimator_.predict(X_test)
        # accuracy = accuracy_score(prediction, y_test)

        return grid.best_params_
