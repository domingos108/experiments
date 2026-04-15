import numpy as np
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import ConvergenceWarning

class PSOMlpRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 pop_size=30, n_gen=100, n_epocas_ft=50,
                 hidden_layer_sizes=(20,), activation='logistic',
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0,
                 use_fitness_sharing=True, sigma_share=0.07, alpha=1.0,
                 random_state=None):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_epocas_ft = n_epocas_ft
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.use_fitness_sharing = use_fitness_sharing
        self.sigma_share = sigma_share
        self.alpha = alpha
        self.random_state = random_state
        
        # O modelo final treinado
        self.best_model_ = None
        self.n_genes_ = None

    def _get_n_genes(self, X, y):
        """Calcula a quantidade total de pesos e biases da MLP."""
        input_dim = X.shape[1]
        output_dim = 1 if len(y.shape) == 1 else y.shape[1]
        
        layers = [input_dim] + list(self.hidden_layer_sizes) + [output_dim]
        n_genes = 0
        for i in range(len(layers) - 1):
            n_genes += layers[i] * layers[i+1] # pesos
            n_genes += layers[i+1]             # biases
        return n_genes

    def _set_weights(self, model, flat_weights):
        """Atribui o vetor plano de pesos ao objeto MLPRegressor do sklearn."""
        shapes = [coef.shape for coef in model.coefs_]
        biases_shapes = [bias.shape for bias in model.intercepts_]
        
        curr = 0
        new_coefs = []
        new_intercepts = []
        
        for shape in shapes:
            size = np.prod(shape)
            new_coefs.append(flat_weights[curr:curr + size].reshape(shape))
            curr += size
        
        for shape in biases_shapes:
            size = np.prod(shape)
            new_intercepts.append(flat_weights[curr:curr + size].reshape(shape))
            curr += size
            
        model.coefs_ = new_coefs
        model.intercepts_ = new_intercepts

    def _fitness_raw(self, weights, X, y, base_model):
        """Avalia o MSE sem o compartilhamento de fitness."""
        self._set_weights(base_model, weights)
        preds = base_model.predict(X)
        return mean_squared_error(y, preds)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.array(X)
        y = y.ravel()
        
        self.n_genes_ = self._get_n_genes(X, y)
        
        # Modelo base apenas para inferência de shapes e fitness rápido
        base_mlp = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, 
                                activation=self.activation, max_iter=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            base_mlp.fit(X[:2], y[:2]) # Inicialização dummy

        # --- Inicialização PSO ---
        pop = rng.uniform(-1, 1, size=(self.pop_size, self.n_genes_))
        velocidades = rng.uniform(-0.1, 0.1, size=(self.pop_size, self.n_genes_))
        
        pbest_pos = pop.copy()
        pbest_scores = np.array([float('inf')] * self.pop_size)
        gbest_pos = None
        gbest_score = float('inf')

        # --- Loop Evolutivo ---
        for gen in range(self.n_gen):
            w = self.w_max - (gen / self.n_gen) * (self.w_max - self.w_min)
            
            # Cálculo de Fitness Bruto
            losses_brutos = np.array([self._fitness_raw(ind, X, y, base_mlp) for ind in pop])
            
            # Fitness Sharing
            if self.use_fitness_sharing:
                # Matriz de distâncias euclidianas entre todos os indivíduos
                dist_matrix = squareform(pdist(pop))
                # m_i = soma das funções de compartilhamento na vizinhança
                sharing_matrix = 1 - (dist_matrix / self.sigma_share) ** self.alpha
                sharing_matrix[dist_matrix >= self.sigma_share] = 0
                m_i = np.sum(sharing_matrix, axis=1)
                losses_calc = losses_brutos * m_i
            else:
                losses_calc = losses_brutos

            # Atualização de memórias
            for i in range(self.pop_size):
                if losses_calc[i] < pbest_scores[i]:
                    pbest_scores[i] = losses_calc[i]
                    pbest_pos[i] = pop[i].copy()
                    
                    if losses_calc[i] < gbest_score:
                        gbest_score = losses_calc[i]
                        gbest_pos = pop[i].copy()

            # Movimento PSO
            r1, r2 = rng.random((2, self.pop_size, self.n_genes_))
            velocidades = (w * velocidades + 
                           self.c1 * r1 * (pbest_pos - pop) + 
                           self.c2 * r2 * (gbest_pos - pop))
            pop += velocidades

        # --- Fine-Tuning (L-BFGS) ---
        # Refina a população final e escolhe o melhor modelo
        best_overall_mse = float('inf')
        
        for individual in pbest_pos:
            model_ft = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                                    activation=self.activation,
                                    solver='lbfgs',
                                    max_iter=self.n_epocas_ft,
                                    random_state=self.random_state)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model_ft.fit(X[:2], y[:2]) # Dummy fit
            
            self._set_weights(model_ft, individual)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model_ft.fit(X, y)
            
            # Avaliação final pós-FT
            final_mse = mean_squared_error(y, model_ft.predict(X))
            if final_mse < best_overall_mse:
                best_overall_mse = final_mse
                self.best_model_ = model_ft
        
        return self

    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("O modelo precisa ser treinado com .fit() antes de prever.")
        return self.best_model_.predict(X)