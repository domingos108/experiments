# 02 — Modelos e Métricas

Este documento detalha as funções de avaliação de erro, a arquitetura base dos modelos e como os experimentos são orquestrados.

---

## 1. Métricas de Avaliação (`src/metrics/metrics.py`)

Todas as métricas operam sobre arrays numpy 1-D e são agregadas pela função `gerenerate_metric_results(y_true, y_pred)`, que retorna um dicionário com todas as chaves abaixo.

### 1.1 Erro Quadrático Médio — MSE

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 1.2 Raiz do Erro Quadrático Médio — RMSE

$$\text{RMSE} = \sqrt{\text{MSE}}$$

É a métrica primária usada para seleção de modelo via validação.

### 1.3 Erro Absoluto Médio — MAE

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 1.4 Erro Percentual Absoluto Médio — MAPE

$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

> Retorna `inf` se qualquer valor real for zero (divisão por zero).

### 1.5 U de Theil

$$U = \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n-1}(\hat{y}_i - \hat{y}_{i+1})^2}$$

Mede o erro relativo à variação das próprias previsões. Valores menores indicam melhor desempenho.

### 1.6 Variância Relativa Média — ARV

$$\text{ARV} = \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2}$$

Compara o erro do modelo com a dispersão das previsões em relação à média real.

### 1.7 Índice de Concordância — IA (Index of Agreement)

$$\text{IA} = 1 - \frac{\sum_{i=1}^{n}|y_i - \hat{y}_i|^2}{\sum_{i=1}^{n}\left(|\hat{y}_i - \bar{y}| + |y_i - \bar{y}|\right)^2}$$

Varia entre 0 e 1; quanto mais próximo de 1, maior a concordância entre previsão e realidade.

### 1.8 Predição de Mudança de Direção — POCID

$$\text{POCID} = \frac{100}{n} \sum_{i=1}^{n-1} \mathbf{1}\left[(y_i - y_{i+1})(\hat{y}_i - \hat{y}_{i+1}) > 0\right]$$

Percentual de acertos na direção (subida/descida) da série. Métrica de acurácia direcional.

### 1.9 Resultado Consolidado

```python
gerenerate_metric_results(y_true, y_pred)
# Retorna:
{
  'MSE':   ...,
  'RMSE':  ...,
  'MAPE':  ...,
  'MAE':   ...,
  'theil': ...,
  'ARV':   ...,
  'IA':    ...,
  'POCID': ...
}
```

---

## 2. Estrutura Base dos Modelos (`src/model/`)

A arquitetura é organizada em três camadas: **utilitários genéricos**, **wrappers de experimento** e **wrapper de busca de hiperparâmetros**.

### 2.1 Utilitários Genéricos (`generics.py`)

| Função / Classe | Responsabilidade |
|-----------------|------------------|
| `fit_predict_ml_schemma` | Loop padrão: `.fit(X_train, y_train)` → `.predict(X_val/X_test)`, com cronometragem. |
| `fit_predict_model` | Orquestrador completo: carrega dados, treina, inverte normalização/diferenciação, calcula métricas. |
| `format_forecats` | Inverte transformações (MinMaxScaler, diferenciação) e computa métricas no espaço original. |
| `save_result` / `open_saved_result` | Serialização de resultados em `.pkl` via `pickle`. |
| `format_names` | Gera caminhos padronizados: `data/result/<experiment_id>/<serie>_<modelo>.pkl`. |
| `ResultExp` | Container simples que encapsula o dicionário `metrics_results`. |

### 2.2 Wrappers de Experimento

Todos os wrappers seguem o mesmo contrato:
- Recebem `model`, `experiment_id`, `base_name`, `model_name`, `experiment_params`.
- Expõem o método `.fit_predict()` que retorna um dicionário `metrics_results`.
- Resultados são salvos automaticamente em `.pkl` se `force=True` ou se ainda não existirem.

#### `SKlearnModel` (`single_ml_model_exp.py`)

Wrapper para qualquer modelo com interface scikit-learn (`fit`/`predict`). Usado para modelos isolados (MLP, SVR, ELM, KAN etc.).

```
exec_config
    │
    ▼
input.open_format_train_val_test()
    │
    ▼
model.fit(X_train, y_train)
    │
    ▼
model.predict(X_val, X_test)
    │
    ▼
inversão de normalização + diferenciação
    │
    ▼
metrics.gerenerate_metric_results()
```

#### `Additive` (`hybrid_system_exp.py`)

Sistema híbrido aditivo do tipo **ARIMA + ML nos resíduos**:

1. Carrega a previsão já salva do modelo linear (ARIMA) via `input_linear_info`.
2. Calcula a série de erros: $e_t = y_t - \hat{y}^{\text{ARIMA}}_t$.
3. Treina o modelo ML para modelar $e_t$.
4. Previsão final: $\hat{y}^{\text{híbrido}}_t = \hat{y}^{\text{ARIMA}}_t + \hat{e}^{\text{ML}}_t$.

```
ARIMA (pré-treinado, .pkl)
    │
    ▼
série de resíduos e_t = y_t - ŷ_ARIMA
    │
    ▼
modelo ML.fit(lags de e_t)
    │
    ▼
ŷ_final = ŷ_ARIMA + ŷ_ML
    │
    ▼
metrics.gerenerate_metric_results()
```

#### `Perturbative` (`perturbative.py`)

Abordagem de ensemble iterativo por perturbação:

1. Treina um modelo base na série original (componente $p_0$).
2. A cada iteração $k$, calcula o erro residual e treina um novo modelo $p_k$ no erro.
3. A previsão final é a soma ponderada das componentes: $\hat{y} = \sum_{k} \lambda_k \cdot p_k$.
4. Suporta `learning_rate` constante ou `invscaling`, early stopping e bagging.

```
modelo_0.fit(série original) → ŷ_0
    │
    ▼
erro_1 = y - ŷ_0
modelo_1.fit(erro_1) → ŷ_1
    │
    ▼
erro_2 = y - (ŷ_0 + λ·ŷ_1)
modelo_2.fit(erro_2) → ŷ_2
    │
    ▼
... (qtd_perturbations iterações)
    │
    ▼
ŷ_final = Σ λ_k · ŷ_k
```

### 2.3 Modelo ARIMA (`arima.py`)

Implementado com `pmdarima.auto_arima` encapsulado na classe `Arima(BaseEstimator)`.

| Parâmetro | Valor |
|-----------|-------|
| `m` | período sazonal vindo do `BASE_INFORMATION` |
| `max_p / max_q` | 5 |
| `max_P / max_Q` | 2 |
| `max_d / max_D` | 2 / 1 |
| Seleção | stepwise automático |

A inferência é feita em modo **walk-forward** (1 passo por vez), com atualização da janela do modelo a cada observação do conjunto de teste via `.update(t)`.

### 2.4 Busca de Hiperparâmetros (`grid_search_exp.py`)

`GridSearch` envolve qualquer wrapper de experimento e realiza busca exaustiva via `sklearn.model_selection.ParameterGrid`.

| Parâmetro | Descrição |
|-----------|-----------|
| `model_class_exp` | Wrapper a ser usado (`SKlearnModel`, `Additive`, `Perturbative`, etc.). |
| `model_parameters` | Dicionário de grade de parâmetros no formato do `ParameterGrid`. |
| `model_exec` | Número de repetições por configuração (para lidar com aleatoriedade). |
| `metric` | Métrica usada para selecionar o melhor modelo (padrão: `RMSE`). |
| `group_metrics_name` | Split usado para seleção (`val_metrics`). |

O melhor conjunto de parâmetros (menor RMSE de validação) é persistido em `.pkl` junto com todos os resultados de execução.

---

## 3. Fluxo Completo de um Experimento

```
1. Configurar experiment_id, base_name, model, experiment_params
        │
        ▼
2. GridSearch._search_params()
   └─ para cada combinação de hiperparâmetros:
        └─ model_class_exp.fit_predict()  (SKlearnModel / Additive / Perturbative)
             ├─ input.open_format_train_val_test()   → dados pré-processados
             ├─ generics.fit_predict_ml_schemma()    → treino + inferência
             ├─ inversão de transformações            → espaço original
             └─ metrics.gerenerate_metric_results()  → dicionário de métricas
        │
        ▼
3. Melhor configuração selecionada por RMSE (validação)
        │
        ▼
4. generics.save_result()  →  data/result/<experiment_id>/<serie>_<modelo>.pkl
        │
        ▼
5. metrics.open_fold_result()  →  leitura e agregação dos .pkl
   └─ retorna df_mean_metrics, df_all_metrics, df_prevs
```

---

## 4. Formato do Dicionário `metrics_results`

Todos os wrappers retornam e persistem o mesmo formato:

```python
metrics_results = {
    'train_predict': np.array([...]),   # previsões no treino (espaço original)
    'val_predict':   np.array([...]),   # previsões na validação
    'test_predict':  np.array([...]),   # previsões no teste
    'val_metrics':   { 'MSE': ..., 'RMSE': ..., ... },  # métricas de validação
    'test_metrics':  { 'MSE': ..., 'RMSE': ..., ... },  # métricas de teste
    'time_exec': {
        'training': float,  # segundos
        'testing':  float
    }
}
```
