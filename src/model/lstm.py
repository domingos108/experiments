import os
import multiprocessing

from sklearn.base import BaseEstimator, RegressorMixin

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

num_cores = multiprocessing.cpu_count()


os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cores)
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_cores)

config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=num_cores,
    inter_op_parallelism_threads=num_cores,
    allow_soft_placement=True,
    device_count={'CPU': num_cores, 'GPU': 0} # Garante que apenas a CPU seja usada, se houver GPU
)

# Define a sessão do Keras/TensorFlow com a configuração
session = tf.compat.v1.Session(config=config)
K.set_session(session)

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
        self.model = Sequential([
        Input(shape=(tw_size, 1)),
        LSTM(units=self.hidden_layer_sizes, activation='relu'),
        Dense(units=1)
        ])

       
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop_callback = EarlyStopping(
            monitor='loss',        # Métrica para monitorar (e.g., 'val_loss' ou 'loss').
                                # 'val_loss' é preferido se você usar validation_split.
            patience=10,           # Número de épocas sem melhora antes de parar.
            restore_best_weights=True, # Restaura os pesos da melhor época.
            verbose=1              # Para ver quando ele interrompe.
        )
        
        self.model.fit(X, 
            y, 
            epochs=self.epochs, 
            batch_size=256, 
            verbose=0, # Mudar para verbose=1 para acompanhar o Early Stopping
            
            # IMPORTANTE: Passar o callback aqui
            callbacks=[early_stop_callback],
            
            # Recomendado: Usar uma fração dos dados para validação (validation_split)
            # e monitorar 'val_loss' no EarlyStopping.
            validation_split=0.1 # Usa 10% dos dados para validação
        )
        return self

    def predict(self, X):
 
        X = X.reshape((X.shape[0], X.shape[1], 1))

        if self.model is not None:
            prev = self.model.predict(X, verbose=0).flatten()
            
            return prev
        else:
            raise RuntimeError("You must train the model before predicting data!")
