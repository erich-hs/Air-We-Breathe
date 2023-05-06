# data_connector.py
# object_connector.py
'''
Actor responsible for transferring data and model binaries around.
'''
class DataConnector:

    def get_data(self):
        raise NotImplemented

    def put_data(self):
        raise NotImplemented
    
class ObjectConnector:

    def get_object(self):
        raise NotImplemented

    def put_object(self):
        raise NotImplemented