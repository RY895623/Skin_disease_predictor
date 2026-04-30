import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    return df