import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

def sigmoid(x):
    # Função de ativação Sigmoid.
    # Transforma o valor de entrada para um valor entre 0 e 1.
    return 1 / (1 + np.exp(-x))


def calculate_h(X, omega, b):
    """
    Calcula a saída de um único neurônio oculto (h_L(X)).
    Assumimos que omega é um vetor coluna e X é uma matriz de entrada.
    """
    # (X @ omega) realiza o produto escalar de cada linha de X com omega
    # O resultado é um vetor de saídas para cada amostra
    return sigmoid(X @ omega + b)

def transform_verify_numpy(array):
    if (
        (isinstance(array, pd.DataFrame)) or 
        (isinstance(array, pd.Series))
        ):
        array = array.to_numpy()

    if (
        (isinstance(array, pd.DataFrame)) or 
        (isinstance(array, pd.Series))
        ):
        array = array.to_numpy()

    if not isinstance(array, np.ndarray):
        raise NotImplementedError("inputs e outputs need to be pd.Dataframe or np.ndarray")
    
    return array

class SCN_III(BaseEstimator, RegressorMixin):

    def __init__(self, 
                t_max: int = 1,
                l_max: int = 10,
                r: int = 0.5,
                lambda_min: float = 0.2,
                lambda_max: float = 1.0,
                lambda_steps: int = 3,
                error_min = 0):
       
        self.t_max = t_max
        self.l_max = l_max
        self.r = r 
        self.error_min = error_min
        self.scaler_x = None
        self.scaler_y = None
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.lambda_steps = lambda_steps
       

    def fit(self, X, y):
        self.lambda_set = np.linspace(self.lambda_min,  self.lambda_max, self.lambda_steps)

        x_train = X.copy()
        y_train = y.copy()
        x_train = transform_verify_numpy(x_train)

        y_train = transform_verify_numpy(y_train)
        if y_train.ndim!=1:
            raise NotImplementedError("multiple target is not implemented")
        
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_x = self.scaler_x.fit(x_train)
        x_train_norm =  self.scaler_x.transform(x_train)

        # TODO: need to be refactored if evolve to multi target
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y.fit(y_train.reshape(-1, 1))
        y_train_norm =  self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()

        self.error_list = []
        input_size = x_train_norm.shape[1]
        e = np.array(y_train_norm.copy())
        
        efro = 1
        r = self.r
        
        n_samples = len(x_train_norm)
        t_max = self.t_max 
        l_max = self.l_max 
        l_i = 1

        self.wstar_list = []
        self.bstar_list = []
        h_l_matrix = np.empty((n_samples, 0))

        while(l_i<=l_max and efro > self.error_min):
            gamma = []
            w = []
            bias = []
            for y in self.lambda_set:
                for k in range(0, t_max):# step 4
                    wl = np.random.uniform(low=-y, high=y, size=input_size)
                    bL = np.random.uniform(low=-y, high=y, size=1)
                    hL = calculate_h(x_train_norm, wl, bL)
                    
                    uL =  (1 - r)/(l_i + 1)
                    eq28 = (np.dot(e.T, hL.T)**2 / np.dot(hL.T, hL)) -  (1 - r - uL) * e.T * e
                    
                    if np.min(eq28) > 0:
                        w.append(wl)
                        bias.append(bL)
                        gamma.append(np.sum(eq28))
                
            if len(w) > 0:
                wstar = w[np.argmax(gamma)]
                bstar = bias[np.argmax(gamma)]
                
                self.wstar_list.append(wstar)
                self.bstar_list.append(bstar)
        
                hlstar = calculate_h(x_train_norm, wstar, bstar)
                h_l_matrix = np.hstack((h_l_matrix, hlstar.reshape(-1, 1)))
            else:
                r = r + np.random.uniform(low=0, high=1 - r , size=1)[0]
                continue
        
            # Garante que T tem 2 dimensões para a multiplicação de matrizes
            T_reshaped = np.array(y_train_norm).reshape(n_samples, -1) if  np.array(y_train_norm).ndim == 1 else y_train_norm
            self.beta_star = np.dot(np.linalg.pinv(h_l_matrix), T_reshaped) 
            #self.beta_star = beta_current#.flatten().tolist() 
        
            el = np.dot(h_l_matrix,  self.beta_star) - T_reshaped
            e = el.copy()
            efro = np.linalg.norm(e, 'fro') / n_samples 
            l_i = l_i + 1
            self.error_list.append(efro )


    def predict(self, X):
        x_test = X.copy()
        x_test = transform_verify_numpy(x_test)
        x_test_norm =  self.scaler_x.transform(x_test)

        n_test_samples, _ = x_test_norm.shape
        #H_test = np.empty((N_test_samples, len(wstar_list))
        h_predict = np.empty((n_test_samples, 0))
        for i, (omega, b) in enumerate(zip(self.wstar_list, self.bstar_list)):
            h_result = calculate_h(x_test_norm, omega, b)
            h_predict = np.hstack((h_predict, h_result.reshape(-1, 1)))

        predictions = np.dot(h_predict,  self.beta_star)
        predictions =  self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions