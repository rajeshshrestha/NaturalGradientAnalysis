import pandas as pd

def get_data():
    '''Load house price data'''
    fields = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','grade','sqft_above','sqft_basement','lat','long','price']
    train_data = pd.read_csv("data/house price/train.csv").fillna(0)[fields]
    train_data = (train_data-train_data.mean())/train_data.std()
    test_data = pd.read_csv("data/house price/test.csv").fillna(0)[fields]
    test_data = (test_data-test_data.mean())/test_data.std()

    return train_data.to_numpy().T, test_data.to_numpy().T
    