import copy
import random
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def get_mlp_weights(mlp, array):
    start_index = 0
    end_index = None
    for mlp_index in range(0, len(mlp.intercepts_)):
        vector_size = len(mlp.intercepts_[mlp_index])
        end_index = start_index + vector_size
        mlp.intercepts_[mlp_index] = array[start_index: end_index]
        start_index = end_index
    
    for mlp_index in range(0, len(mlp.coefs_[0])):
        vector_size = len(mlp.coefs_[0][mlp_index])
        end_index = start_index + vector_size
        mlp.coefs_[0][mlp_index] = array[start_index: end_index]
        start_index = end_index
    
    for mlp_index in range(0, len(mlp.coefs_[1])):
        vector_size = len(mlp.coefs_[1][mlp_index])
        end_index = start_index + vector_size
        mlp.coefs_[1][mlp_index] = array[start_index: end_index]
        start_index = end_index

    return mlp

def metric_models(mlps, X, y):
    predicts = [m.predict(X) for m in mlps]

    mses = [mean_square_error(y, p) for p in predicts]

    return mses
    
def start_population(
        hidden_layer_sizes, 
        pop_size, 
        dimension_size,
        upper_bound, 
        lower_bound,
        X, 
        y
    ):
    
    positions = (
        np.random.rand(pop_size, dimension_size) * 
        (upper_bound - lower_bound) + 
        lower_bound)
    
    # only for create a coef e intercepts_
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_sizes, ), 
        max_iter=1000,
        solver = 'lbfgs'
    )
    mlp = mlp.fit(X, y)

    mlps = [get_mlp_weights(copy.deepcopy(mlp),  p) for p in positions]

    return mlps, positions

def selecao_roleta(populacao, mses):

    fitness_vals = [1.0 / (m + 1e-6) for m in mses]
    
    total_fit = sum(fitness_vals)
    pais = []
    for _ in populacao:
        pick = random.uniform(0, total_fit)
        current = 0
        for ind, fit in zip(populacao, fitness_vals):
            current += fit
            if current >= pick:
                pais.append(ind)
                break
    return pais

def selecao_aleatoria(population, metrics):
        return [random.choice(population) for _ in population]

def parents_selection(population, metrics, crossover_type):

    if crossover_type == "tournament":
        #pais = tournament_selection(population, metrics)
        pass
    elif crossover_type == "roulette":
        pais = selecao_roleta(population, metrics)

    elif crossover_type == "random":
        pais = selecao_aleatoria(population, metrics)

    return pais

def mutation(individuo, meta_heuristic_params):
    lower_bound = meta_heuristic_params['lower_bound']
    upper_bound = meta_heuristic_params['upper_bound']
    pct_mutation = meta_heuristic_params['pct_mutation']
    
    return [
        np.clip(gene + random.uniform(-0.1, 0.1), lower_bound, upper_bound) if random.random() < pct_mutation else gene
        for gene in individuo
    ]

def crossover_exec(parents1, parents2, meta_heuristic_params):
    dimension_size = meta_heuristic_params['dimension_size']
    ponto = random.randint(1, dimension_size - 1)
    child1 = np.concatenate((parents1[:ponto], parents2[ponto:]))
    child2 = np.concatenate((parents2[:ponto], parents1[ponto:]))
    return child1, child2

def crossover(parents, meta_heuristic_params):
    nova_pop = []
    for i in range(0, len(parents), 2):
        if i+1 < len(parents):
            f1, f2 = crossover_exec(
                parents[i],
                parents[i+1], 
                meta_heuristic_params
            )

            nova_pop.extend([
                mutation(f1, meta_heuristic_params), 
                mutation(f2, meta_heuristic_params)
                ])
        else:
            nova_pop.append(mutation(parents[i], meta_heuristic_params))

    return nova_pop

def ga_iteration(population, metrics, meta_heuristic_params):
    parents = parents_selection(
        population, 
        metrics, 
        meta_heuristic_params['crossover_type']
    )
    
    nova_pop = crossover(parents, meta_heuristic_params)

    return nova_pop

def force_to_numpy(data):
    #1. Check if it's a Pandas DataFrame.
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    
    #1. Check if it's a NumPy Array
    elif isinstance(data, np.ndarray):
        return data
    
    #3. If it's anything else, it raises an error.
    else:
        raise TypeError(
            f"Invalid data type: {type(data)}. "
            "The variable must be a Pandas DataFrame or a NumPy Array."
        )
    
class MetaHeuristicMLP(BaseEstimator, RegressorMixin):

    def __init__(
            self,
            iterations: int = 100,
            crossover_type: str = 'random',
            pop_size: int = 100, 
            pct_mutation: float = 0.01, 
            hidden_layer_sizes: int = 100
        ): 
        
        self.upper_bound = 1
        self.lower_bound = -1

        self.iterations = iterations
        self.pop_size = pop_size 
        self.hidden_layer_sizes = hidden_layer_sizes
        self.crossover_type = crossover_type
        self.final_mlp = None
        self.pct_mutation = pct_mutation

    def fit(self, X, y):
        self.meta_heuristic_params = {
            'crossover_type': self.crossover_type,
            'pct_mutation': self.pct_mutation,
            'upper_bound': self.upper_bound,
            'lower_bound': self.lower_bound,
        }

        self.X = force_to_numpy(X)
        self.y = force_to_numpy(y)
        input_size = self.X.shape[1]
        
        if self.y.ndim > 1:
            output_size = y.shape[1]

            if output_size > 1:
                raise Exception("Multiple outputs not allowed")
            
            self.y = self.y.flatten()
        else:
            output_size = 1

        self.dimension_size =( 
            (input_size * self.hidden_layer_sizes) + 
            (self.hidden_layer_sizes * output_size) + 
            self.hidden_layer_sizes + 
            output_size
        )
        
        self.meta_heuristic_params['dimension_size'] = self.dimension_size

        mlps, population = start_population(
            self.hidden_layer_sizes,
            self.pop_size, 
            self.dimension_size, 
            self.upper_bound, 
            self.lower_bound, 
            self.X, 
            self.y
        )

        self.best_fitness = []
        best_individuals = []
        for geracao in tqdm(range(self.iterations)):

            metrics = metric_models(mlps, self.X, self.y)
            best_arg = np.argmin(metrics)

            self.best_fitness.append(metrics[best_arg])
            best_individuals.append(
                population[best_arg]
            )

            population = ga_iteration(population, metrics, self.meta_heuristic_params)

            mlps = [get_mlp_weights(copy.deepcopy(mlps[0]),  p) for p in population]
        print(f'best arg {np.argmin(self.best_fitness)}')
        self.selected_individual = best_individuals[np.argmin(self.best_fitness)]
        self.final_mlp = get_mlp_weights(
            copy.deepcopy(mlps[0]),  
            self.selected_individual
        )
        self.final_mlp.fit(self.X, self.y)
    
    def predict(self, X):
        X_predict = force_to_numpy(X)
        if self.final_mlp is None:
            raise Exception(
                "It is necessary to run the .fit command before the predict command."
            )
        return self.final_mlp.predict(X_predict)