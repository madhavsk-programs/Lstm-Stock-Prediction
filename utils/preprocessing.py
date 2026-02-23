import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATH,WINDOW_SIZE,TRAIN_SPLIT

class TimeSeriesPreprocessor:
    def __init__(self):
        self.scaler=MinMaxScaler(feature_range=(0,1))
        self.data=None
        self.scaled_data=None

    def load_data(self):
        self.data=pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully!")
        print("Shape:",self.data.shape)
        print(self.data.head())
        return self.data
    
    def preprocess_data(self):
        self.data['date']=pd.to_datetime(self.data['date'])
        self.data=self.data.sort_values('date')
        close_prices=self.data[['close']].values

        print("Selected Feature: Close Price")
        print("Progonal Data shape:",close_prices.shape)

        self.scaled_data=self.scaler.fit_transform(close_prices)
        return self.scaled_data
    
    def create_sequences(self):
        X=[]
        y=[]

        for i in range(WINDOW_SIZE, len(self.scaled_data)):
            X.append(self.scaled_data[i-WINDOW_SIZE:i])
            y.append(self.scaled_data[i])

        X = np.array(X)
        y = np.array(y)
        print("Sequence Creation Completed!")
        print("X Shape:", X.shape)
        print("y Shape:", y.shape)

        return X, y
    def train_test_split(self, X, y):
        train_size = int(len(X) * TRAIN_SPLIT)

        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        print("Training Samples:", X_train.shape)
        print("Testing Samples:", X_test.shape)

        return X_train, X_test, y_train, y_test

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
