# Class for data loading
from scipy import io
import pickle
class Loader:
    def __init__(self, path):
        self.data_path = path
    
    def load_one_rec(self, filename):
        one_record = io.loadmat(f'{self.data_path}/{filename}.mat')
        ecg_one_rec = one_record['ECG'][0,0][2]     # 12 lead ECG data
        return ecg_one_rec
    
    def load_batch(self, rec_names, batch_size = 32):
        # i = 0
        num_batches = len(rec_names) // batch_size + 1
        for batch in range(num_batches):
            data = []
            for rec in rec_names[batch_size*batch : batch_size * (batch+1)]:
                data.append(self.load_one_rec(rec))

            yield data
    
    def load_model(self, model_name):
        with open(self.data_path+model_name, 'rb') as f:
            model = pickle.load(f)
        
        return model