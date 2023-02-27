import pandas as pd
import numpy as np

def get_data(test_proportion):
    '''Load ecoli data'''
    features = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']
    fields = features+['class']
    ecoli_data = pd.read_csv("data/ecoli/ecoli.data", names=['sequence_name']+features+["class"]).fillna(0)[fields]

    '''Normalize continuous fields'''
    continuous_fields = features
    if continuous_fields:
        ecoli_data[continuous_fields] = ecoli_data[continuous_fields].replace("?", 0)
        ecoli_data[continuous_fields] = (ecoli_data[continuous_fields] - ecoli_data[continuous_fields].mean()) / ecoli_data[continuous_fields].std()

    '''One hot encoding of the categorical fields'''
    categorical_fields = list(set(features) - set(continuous_fields))
    if categorical_fields:
        ecoli_data[categorical_fields] = ecoli_data[categorical_fields].replace("?", "Unknown")
        ecoli_data = pd.concat([ pd.get_dummies(ecoli_data[categorical_fields]), ecoli_data], axis=1).drop(categorical_fields, axis=1)

    '''Encode the classes'''
    class_dict = {'cp':0, 'im':1, 'pp':1, 'imU': 1, 'om':1, 'omL': 1, 'imL': 1, 'imS':1}
    ecoli_data['class'] = ecoli_data['class'].map(class_dict)

    '''Indices for train and test data'''
    data_indices = set(ecoli_data.index)
    test_indices = set(np.random.choice(list(data_indices), size=int(
        len(data_indices)*test_proportion), replace=False))
    train_indices = data_indices - test_indices

    train_data, test_data = ecoli_data.loc[list(train_indices)].to_numpy(
    ).T, ecoli_data.loc[list(test_indices)].to_numpy().T
    
    return train_data, test_data


if __name__ == "__main__":
    get_data(0.2)