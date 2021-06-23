from read_write import Loader, Writer
from preprocessing import Extractor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def get_started():
    path = './TrainingSet/'
    labels = pd.read_csv('REFERENCE.csv')
    fs = 500
    
    loader_obj = Loader(path)
    extractor_obj = Extractor(fs)

    batches = loader_obj.load_batch(labels.Recording)
    
    
    numerical_features = []
    wavelet_features = []
    hos_features = []
    lbp_features = []
    y = []
    y_idx = 0

    print('Extracting features, might take a while...')
    for batch_data in batches:
        # bactch_data is a list of 12-lead ECG records
        for rec in batch_data:
            try:
                features = extractor_obj.get_features(rec[3])   # 3rd index is for aVr lead
                numerical_features.append(features[:8])
                wavelet_features.append(features[8])
                lbp_features.append(features[9])
                hos_features.append(features[10])
                y.append(extractor_obj.to_categorical(labels.First_label[y_idx]-1, num_classes=9))
            except:
                continue
            y_idx += 1


    X = np.hstack(tuple(map(np.array, [numerical_features, wavelet_features, hos_features])))
    y = np.array(y)

    print('Done extracting.')
    print('Training the classifier...')
    classifier = Pipeline([
                    ('std', StandardScaler()),
                    ('gbm', OneVsRestClassifier(GradientBoostingClassifier(), n_jobs=1))
                ])
    classifier.fit(X, y)

    writer_obj = Writer()
    writer_obj.write_an_obj(classifier, 'models/gbm_model.pkl')
    
    print('Saved model to the disk')
    


if __name__ == '__main__':
    get_started()