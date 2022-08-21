# the clustering model

from tensorflow.keras import models

class Cluster:

    def __init__(self, save_dir: str):
        self.model = models.load_model(save_dir)

    def predict(self, X):
        return self.model.predict(X)
