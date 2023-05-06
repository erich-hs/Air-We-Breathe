# inference.py
'''
The inference application will follow the following steps:
Get the data with the DataConnector
Load the model with the TrainingPipeline (we will need to add that method)
Predict on that data with the TrainingPipeline
Put the predictions back into the database
'''
from data.connector import DataConnector
from ml.trainer import TrainingPipeline

def run():

    connector = DataConnector()
    pipeline = TrainingPipeline()
    data = connector.get_data()
    predictions = pipeline.predict(data)
    connector.put_data(predictions)

if __name__ == '__main__':
    run()