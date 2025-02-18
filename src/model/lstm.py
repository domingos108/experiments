from sklearn.base import BaseEstimator, RegressorMixin

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras import backend as K


class CustomLSTM(BaseEstimator, RegressorMixin):

    def __init__(self, hidden_layer_sizes=50,
                 epochs=1000):
        K.clear_session()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.model = None


    def fit(self, X, y=None):
        tw_size = X.shape[1]
        X = X.reshape((X.shape[0], X.shape[1], 1))
 
        self.model = Sequential()
        self.model.add(Input(shape=(tw_size, 1))) 
        self.model.add(LSTM(self.hidden_layer_sizes, return_sequences=True)) # No input_shape here
        self.model.add(LSTM(self.hidden_layer_sizes))  
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
        return self

    def predict(self, X):
 
        X = X.reshape((X.shape[0], X.shape[1], 1))

        if self.model is not None:
            prev = self.model.predict(X, verbose=0).flatten()
            
            return prev
        else:
            raise RuntimeError("You must train the model before predicting data!")
