import utils
from sklearn.cluster import KMeans



class processor():

    def __init__(self,vocab_size=30):
        self._kmeans=None
        self.vocabulary=vocab_size


    @property
    def kmeans(self):
        return self._kmeans
    @kmeans.setter
    def kmeans(self,X):
        self._kmeans=KMeans(n_clusters=self.vocabulary, random_state=0).fit(X)