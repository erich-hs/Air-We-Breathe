# model.py
'''
Learning the parameters of the model
the Model will take care of that with the fit() method.
For inferring, we use the predict() method.
'''
from data.connector import ObjectConnector

class Model:

    def set_params(self, params):
        raise NotImplemented

    def fit(self, X, y):
        raise NotImplemented

    def predict(self, X):
        raise NotImplemented

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)
    
    # Save at and load from the model registry using the ObjectConnector
    # We assume that the self.net attribute captures the Model state
    def save(self):
        connector = ObjectConnector()
        connector.put_object(self.net, '/path/to/model/registry')
    
    def load(self):
        connector = ObjectConnector()
        self.net = connector.get_object('/path/to/model/registry')