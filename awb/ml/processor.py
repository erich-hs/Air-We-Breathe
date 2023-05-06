# data_processor.py
'''
Preprocess the data for ML model
'''
from data.connector import ObjectConnector

class DataProcessor:

    def fit(self, data):
        raise NotImplemented
        return self

    def transform(self, data):
        raise NotImplemented

    def fit_transform(self, data):
        return self.fit(data).transform(data)
    
    # Save at and load from the model registry using the ObjectConnector
    # We assume that the self.params attribute captures the DataProcessor state
    def save(self):
        connector = ObjectConnector()
        connector.put_object(self.params, '/path/to/model/registry')
    
    def load(self):
        connector = ObjectConnector()
        self.net = connector.get_object('/path/to/model/registry')