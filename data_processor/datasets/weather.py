import pandas as pd
import numpy as np

def get_data(test_proportion):
    '''Load weather data'''
    weather_data = pd.read_csv("data/weather/weatherHistory.csv")[["Apparent Temperature (C)",
                                                                "Humidity",
                                                                "Wind Speed (km/h)",
                                                                "Wind Bearing (degrees)",
                                                                "Visibility (km)",
                                                                "Pressure (millibars)",
                                                                "Temperature (C)"]]
    weather_data = (weather_data - weather_data.mean()) / weather_data.std()

    '''Indices for train and test data'''
    data_indices = set(weather_data.index)
    test_indices = set(np.random.choice(list(data_indices), size=int(
        len(data_indices)*test_proportion), replace=False))
    train_indices = data_indices - test_indices

    train_data, test_data = weather_data.loc[list(train_indices)].to_numpy(
    ).T, weather_data.loc[list(test_indices)].to_numpy().T
    
    return train_data, test_data
