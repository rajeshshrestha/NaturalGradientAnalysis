import pandas as pd
import numpy as np

def get_data(test_proportion):
    '''Load cancer data'''
    features = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    fields = features+['class']
    cancer_data = pd.read_csv("data/cancer/breast-cancer.data", names=['class']+features).fillna(0)[fields]

    '''Normalize continuous fields'''
    continuous_fields = []
    if continuous_fields:
        cancer_data[continuous_fields] = cancer_data[continuous_fields].replace("?", 0)
        cancer_data[continuous_fields] = (cancer_data[continuous_fields] - cancer_data[continuous_fields].mean()) / cancer_data[continuous_fields].std()

    '''One hot encoding of the categorical fields'''
    categorical_fields = list(set(features) - set(continuous_fields))
    if categorical_fields:
        cancer_data[categorical_fields] = cancer_data[categorical_fields].replace("?", "Unknown")
        cancer_data = pd.concat([ pd.get_dummies(cancer_data[categorical_fields]), cancer_data], axis=1).drop(categorical_fields, axis=1)

    '''Encode the classes'''
    class_dict = {'no-recurrence-events':0, 'recurrence-events':1}
    cancer_data['class'] = cancer_data['class'].map(class_dict)

    '''Indices for train and test data'''
    data_indices = set(cancer_data.index)
    test_indices = set(np.random.choice(list(data_indices), size=int(
        len(data_indices)*test_proportion), replace=False))
    train_indices = data_indices - test_indices

    train_data, test_data = cancer_data.loc[list(train_indices)].to_numpy(
    ).T, cancer_data.loc[list(test_indices)].to_numpy().T
    
    return train_data, test_data


if __name__ == "__main__":
    get_data(0.2)