# training.py
'''
The training application will pass through the following steps:
Get data with the DataConnector
Train with the TrainingPipeline
Save the model (we will need to add a save method later).
'''
from data.connector import DataConnector
from ml.trainer import TrainingPipeline

def run():
    
    connector = DataConnector()
    pipeline = TrainingPipeline()
    data = connector.get_data()
    pipeline.train(data)

if __name__ == '__main__':
    run()