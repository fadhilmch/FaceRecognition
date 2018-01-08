import os
import pickle
import numpy as np
from sklearn import neighbors, svm

BASE_DIR = os.path.dirname(__file__) + '/'
PATH_TO_PKL = 'classifier.pkl'


class FaceClassifier:
    def __init__(self, model_path=None):

        self.model = None
        if model_path is None:
            return
        elif model_path == 'default':
            model_path = BASE_DIR+PATH_TO_PKL

        # Load models
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def train(self, X, y, model='knn', save_model_path=None):
        if model == 'knn':
            self.model = neighbors.KNeighborsClassifier(3, weights='uniform')
        else:  # svm
            self.model = svm.SVC(gamma=0.001, C=100, kernel='linear')
        self.model.fit(X, y)
        if save_model_path is not None:
            with open(save_model_path, 'wb') as f:
                pickle.dump(self.model, f)

    def classify(self, descriptor):
        if self.model is None:
            print('Train the model before doing classifications.')
            return

        return self.model.predict(descriptor)
