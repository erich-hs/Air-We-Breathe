# training_pipeline.py
'''
Handles the logic between processor.py, model.py, and tuner.py
Implements methods to fit the model (including hyperparameter tuning) with .fit()
and inference from the model with .predict()
'''
from ml.processor import DataProcessor
from ml.tuner import CrossValidator
from ml.model import Model

class TrainingPipeline:

    def __init__(self):
        self.model = Model()
        self.processor = DataProcessor()
        self.tuner = CrossValidator(self.model)

    def train(self, data):
        '''
        * Fit and transform the data
        * Cross validate hyperparameters
        * Train the model
        * Save the model and processor (if trainable - such as vectorizer/etc)
        '''
        # Fit and transform the data
        X, y = self.processor.fit_transform(data)

        # Find the best hyperparameters for the model
        best_params = self.tuner(X, y)

        # Set the model with best hyperparameters
        self.model.set_params(best_params)

        # Fit the model
        self.model.fit(X, y)

        # Save the pipeline
        self.save()

    def predict(self, data):
        '''
        * Load the model
        * Transform the data
        * Predict on that data
        '''
        # Load the pipeline
        self.load()

        # Transform the data
        X, _ = self.processor.transform(data)

        # Return the predictions
        return self.model.predict(X)
    
    # The save and load methods are delegated to the Model and DataProcessor
    # They can locally deal with the file types for the different objects to be saved
    def save(self):
        '''
        Save pipeline
        '''
        self.model.save()
        self.processor.save()
    
    def load(self):
        '''
        Load the pipeline
        '''
        self.model.load()
        self.processor.load()