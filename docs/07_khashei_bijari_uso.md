# Exemplo de Uso: KhasheiBijariHybrid

## Arquitetura de Khashei & Bijari (2011)

A classe `KhasheiBijariHybrid` implementa a arquitetura híbrida não-linear proposta por Khashei & Bijari (2011), que difere da abordagem aditiva simples de Zhang.

### Diferença chave entre as arquiteturas

**Zhang (Additive)**:
```
ŷ_t = L̂_t + N̂_t
```
onde treina-se separadamente:
1. ARIMA → L̂_t (componente linear)
2. MLP(e_{t-1}, ..., e_{t-n}) → N̂_t (componente não-linear dos resíduos)
3. Soma final

**Khashei & Bijari (NonLinear)**:
```
y_t = f(e_{t-1}, ..., e_{t-n1}, L̂_t, z_{t-1}, ..., z_{t-m1})
```
onde treina-se uma única rede neural que recebe:
1. Lags dos resíduos ARIMA: e_{t-1}, ..., e_{t-n1}
2. Previsão linear no tempo t: L̂_t
3. Lags da série original: z_{t-1}, ..., z_{t-m1}
4. Output direto: y_t (não há soma)

---

## Exemplo 1: Uso básico em notebook

```python
from sklearn.neural_network import MLPRegressor
from model.hybrid_system_exp import KhasheiBijariHybrid
from model.grid_search_exp import GridSearch

# 1. Definir o modelo ML base
mlp_model = MLPRegressor(
    hidden_layer_sizes=(50, 25),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# 2. Configurar parâmetros do experimento
experiment_params = {
    'linear_model_name': 'arima',           # Nome do ARIMA pré-treinado
    'n1_lags': 3,                           # Lags dos resíduos ARIMA
    'm1_lags': 5,                           # Lags da série original
    'test_size': 0.1,
    'val_size': 0.1,
    'horizon': 1,
    'lag_size': 5,                          # Usado para carregamento de dados
    'diff_kpss': True
}

# 3. Instanciar e executar
kb_hybrid = KhasheiBijariHybrid(
    model=mlp_model,
    experiment_id='chamados',
    base_name='airlines.txt',
    model_name='kb_mlp',
    normalize=True,
    experiment_params=experiment_params
)

kb_hybrid.fit_predict()

# 4. Acessar resultados
print("RMSE Teste:", kb_hybrid.metrics_results['test_metrics']['RMSE'])
print("MAE Teste:", kb_hybrid.metrics_results['test_metrics']['MAE'])
```

---

## Exemplo 2: Grid Search para otimização

```python
from sklearn.neural_network import MLPRegressor
from model.hybrid_system_exp import KhasheiBijariHybrid
from model.grid_search_exp import GridSearch

# Modelo base
mlp = MLPRegressor(max_iter=500, random_state=42)

# Grade de hiperparâmetros
param_grid = {
    'hidden_layer_sizes': [(25,), (50,), (50, 25), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],  # regularização L2
    'n1_lags': [3, 5, 7],            # Lags dos resíduos
    'm1_lags': [3, 5, 7, 10]         # Lags da série original
}

# Parâmetros fixos
fixed_params = {
    'linear_model_name': 'arima',
    'test_size': 0.1,
    'val_size': 0.1,
    'horizon': 1,
    'lag_size': 7,
    'diff_kpss': True
}

# Grid Search
gs = GridSearch(
    model=mlp,
    model_class_exp=KhasheiBijariHybrid,
    model_parameters=param_grid,
    experiment_id='chamados',
    base_name='airlines.txt',
    model_name='kb_mlp_gs',
    experiment_params=fixed_params,
    model_exec=3,              # 3 repetições por configuração
    metric='RMSE',
    group_metrics_name='val_metrics'
)

gs.execution()
```

---

## Exemplo 3: Comparação Additive vs Khashei-Bijari

```python
import pandas as pd
from sklearn.neural_network import MLPRegressor
from model.hybrid_system_exp import Additive, KhasheiBijariHybrid

# Configuração comum
mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
base_name = 'airlines.txt'
experiment_id = 'comparison'

common_params = {
    'linear_model_name': 'arima',
    'test_size': 0.1,
    'val_size': 0.1,
    'horizon': 1,
    'lag_size': 5,
    'diff_kpss': True
}

# 1. Additive (Zhang)
additive = Additive(
    model=mlp,
    experiment_id=experiment_id,
    base_name=base_name,
    model_name='zhang_mlp',
    experiment_params=common_params
)
additive.fit_predict()

# 2. Khashei-Bijari
kb_params = {**common_params, 'n1_lags': 5, 'm1_lags': 5}
kb = KhasheiBijariHybrid(
    model=mlp,
    experiment_id=experiment_id,
    base_name=base_name,
    model_name='kb_mlp',
    experiment_params=kb_params
)
kb.fit_predict()

# 3. Comparação
results = pd.DataFrame({
    'Modelo': ['ARIMA-MLP (Zhang)', 'ARIMA-MLP (Khashei-Bijari)'],
    'RMSE': [
        additive.metrics_results['test_metrics']['RMSE'],
        kb.metrics_results['test_metrics']['RMSE']
    ],
    'MAE': [
        additive.metrics_results['test_metrics']['MAE'],
        kb.metrics_results['test_metrics']['MAE']
    ],
    'MAPE': [
        additive.metrics_results['test_metrics']['MAPE'],
        kb.metrics_results['test_metrics']['MAPE']
    ]
})

print(results.to_string(index=False))
```

---

## Exemplo 4: Integração com filtro de não-linearidade BDS

```python
from utils.nonlinearity_test import test_nonlinearity_report
from model.hybrid_system_exp import KhasheiBijariHybrid
import pandas as pd

# Pré-requisito: ARIMA já treinado e salvo
# Calcular resíduos do ARIMA para teste BDS
from model.hybrid_system_exp import input_linear_info

experiment_params = {
    'linear_model_name': 'arima',
    'test_size': 0.1,
    'val_size': 0.1,
    'horizon': 1,
    'lag_size': 5,
    'diff_kpss': True,
    'n1_lags': 5,
    'm1_lags': 5
}

# Carregar resíduos
error_series, ts_forecast, base_info, _ = input_linear_info(
    experiment_id='chamados',
    base_name='airlines.txt',
    experiment_params=experiment_params
)

# Teste de não-linearidade
bds_report = test_nonlinearity_report(pd.Series(error_series))

print("=== Teste BDS ===")
print(f"Decisão: {bds_report['decision']}")
print(f"P-value mínimo: {bds_report['min_p_value']:.6f}")

# Decidir se usar híbrido
if bds_report['has_nonlinearity']:
    print("\n[✓] Não-linearidade detectada → Treinar Khashei-Bijari")
    
    mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    kb = KhasheiBijariHybrid(
        model=mlp,
        experiment_id='chamados',
        base_name='airlines.txt',
        model_name='kb_mlp',
        experiment_params=experiment_params
    )
    kb.fit_predict()
    
    print(f"RMSE híbrido: {kb.metrics_results['test_metrics']['RMSE']:.4f}")
else:
    print("\n[!] Resíduos são ruído branco → Usar apenas ARIMA")
    # Carregar métricas do ARIMA já salvo
```

---

## Parâmetros importantes

### `n1_lags` (lags dos resíduos)
- **Padrão**: `lag_size` do experimento
- **Recomendação**: testar 3, 5, 7 via Grid Search
- Controla quantos lags passados dos resíduos ARIMA são usados

### `m1_lags` (lags da série original)
- **Padrão**: `lag_size` do experimento
- **Recomendação**: testar 3, 5, 7, 10 via Grid Search
- Pode ser diferente de `n1_lags` para capturar padrões de diferentes escalas temporais

### `normalize`
- **Padrão**: `True`
- Aplica MinMaxScaler apenas no conjunto de treino (sem data leakage)

### `linear_model_name`
- **Obrigatório**
- Nome do modelo ARIMA pré-treinado (ex: 'arima', '1arima')

---

## Notas para dissertação

1. **Citação correta**: Khashei, M., & Bijari, M. (2011). "A novel hybridization of artificial neural networks and ARIMA models for time series forecasting." Applied Soft Computing, 11(2), 2664-2675.

2. **Diferença arquitetural**: Ressaltar que a entrada combinada permite à rede neural aprender interações não-lineares entre a previsão ARIMA e os padrões da série original, algo que a soma aditiva não permite.

3. **Experimento comparativo**: Rodar baseline com `Additive` (Zhang) vs `KhasheiBijariHybrid` nas mesmas 17 séries para validar qual arquitetura performa melhor.

4. **Análise de hiperparâmetros**: Documentar influência de `n1_lags` e `m1_lags` no desempenho final — séries com forte sazonalidade podem se beneficiar de `m1_lags` maiores.
