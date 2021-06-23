import pickle

class Writer:
    def write_an_obj(self, obj, path):
        '''
        Write an object as a binary file to disk on the path provided
        '''
        with open(path, 'wb') as f:
            pickle.dump(obj, f)