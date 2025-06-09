import pandas as pd
import os

def load_water_data():
    """
    Load the water quality dataset and return the input features and target labels.
    
    Returns:
        x (numpy.ndarray): Input features.
        y (numpy.ndarray): Target labels.
    """
    
    # get location of file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(current_dir, 'water_potability.csv'))

    # remove missing values
    data = data.dropna(axis=0, how='any')

    # convert into inputs and outputs and cast into numpy arrays
    x = data.drop('Potability', axis=1).to_numpy()
    y = data['Potability'].to_numpy()

    print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns.')
    print(f'The dataset has {data.isna().sum().sum()} null values.')

    # remove missing values
    data = data.dropna(axis=0, how='any')
    data.describe()

    # convert into inputs and outputs and cast into numpy arrays
    x = data.drop('Potability', axis=1).to_numpy()
    y = data['Potability'].to_numpy()

    print(f'The input data has shape {x.shape} and the output data has shape {y.shape}.')
    print(f'the first sample is {x[0]} and the output is {y[0]}.')

    print(f'The fraction of positive samples is {y.mean():.2f}.')
    
    return x, y


if __name__ == "__main__":
    load_water_data()