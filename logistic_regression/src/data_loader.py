import pandas as pd
import numpy as np

def load_train_data(path='data/train.csv'):
    return pd.read_csv(path)