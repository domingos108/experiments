# 04 — Experimentos e Sistemas Híbridos

Este documento descreve o fluxo de execução prático dos experimentos, a diferença entre modelo único e sistema híbrido residual, e um guia passo a passo para reproduzir os resultados.

---

## 1. Dois Paradigmas de Modelagem

O projeto implementa dois paradigmas fundamentais que diferem matematicamente e arquiteturalmente:

| Aspecto | Modelo Único | Sistema Híbrido Residual (ARIMA + ML) |
|---------|-------------|---------------------------------------|
| **Entrada do ML** | Lags da série original $y_t$ | Lags da série de resíduos $e_t = y_t - \hat{y}^{\text{ARIMA}}_t$ |
| **Saída** | $\hat{y}_t$ diretamente | $\hat{y}_t = \hat{y}^{\text{ARIMA}}_t + \hat{e}^{\text{ML}}_t$ |
| **Hipótese** | ML captura toda a estrutura da série | ARIMA captura parte linear; ML captura não-linearidade residual |
| **Dependência** | Nenhuma (execução independente) | ARIMA **deve** ser executado antes |
| **Notebook** | `single_models/mlp_exec.ipynb` | `residual_hydridsystem/arima_mlp.ipynb` |

---

## 2. Arquitetura: Modelo Único (`single_models/`)

### 2.1 Fluxo de Execução

```
Célula 1: Configuração
  ├─ Instanciar MLPRegressor (ou SVR, ELM, KAN, etc.)
  ├─ Definir experiment_id, model_name, normalize, force
  ├─ Definir experiment_params  →  { diff_kpss, horizon, type_filter }
  └─ Definir model_parameters   →  grade de hiperparâmetros
         │
         ▼
grid_search_exp.grid_seach_multiple_bases(
    SKlearnModel, model, normalize,
    model_parameters, experiment_params,
    model_exec, model_name, experiment_id
)
         │
         ▼
Para cada base em config.BASE_NAME_LIST:
    └─ GridSearch.execution()
         ├─ Para cada combinação em ParameterGrid(model_parameters):
         │     └─ Para cada repetição em range(model_exec):
         │           ├─ input.open_format_train_val_test()
         │           ├─ model.fit(X_train, y_train)
         │           ├─ model.predict(X_val, X_test)
         │           ├─ inverter normalização/diferenciação
         │           └─ metrics.gerenerate_metric_results()
         │
         ├─ Selecionar melhor config por RMSE (validação)
         └─ Salvar → data/result/<experiment_id>/<serie>_1<model_name>.pkl
```

### 2.2 Formulação Matemática

O modelo recebe a janela de lags e prevê diretamente o próximo valor:

$$\hat{y}_{t+h} = f_{\text{ML}}\left(y_{t}, y_{t-1}, \ldots, y_{t-p}\right)$$

Onde $p$ é o `lag_size` e $h$ o `horizon`. Se `diff_kpss=True` e a série for não estacionária, opera sobre a série diferenciada e reverte ao final.

### 2.3 Configurações do `experiment_params`

```python
experiment_params = {
    'diff_kpss': True,       # aplica diferenciação sazonal se necessário
    'horizon':   1,          # passos à frente
    'type_filter': None      # None | 'ma' | 'db4'
}
```

---

## 3. Arquitetura: Sistema Híbrido Residual (`residual_hydridsystem/`)

### 3.1 Dependência: ARIMA deve ser executado primeiro

O sistema híbrido lê a previsão salva do ARIMA para calcular os resíduos. **Sem o `.pkl` do ARIMA, o notebook híbrido falha.**

```
notebook/single_models/arima_exec.ipynb   ←  DEVE SER EXECUTADO PRIMEIRO
    └─ data/result/<experiment_id>/<serie>_1arima.pkl
```

### 3.2 Fluxo de Execução (`arima_mlp.ipynb`)

```
Célula 1: Importações
  ├─ MLPRegressor, hybrid_system_exp, grid_search_exp
  └─ %autoreload 2

Célula 2: Configuração e execução
  ├─ Instanciar MLPRegressor(activation='identity')
  ├─ experiment_params['linear_model_name'] = '1arima'   ← nome do .pkl do ARIMA
  └─ grid_search_exp.grid_seach_multiple_bases(
         hybrid_system_exp.Additive, ...
     )
         │
         ▼
Para cada base em config.BASE_NAME_LIST:
    └─ Additive.fit_predict()
         ├─ hybrid_system_exp.input_linear_info()
         │     ├─ Carregar .pkl do ARIMA  →  ts_forecast
         │     └─ e_t = y_t - ŷ_ARIMA         ← série de resíduos
         │
         ├─ input.open_format_train_val_test(pd.Series(e_t), ...)
         │     └─ janelamento dos resíduos → X_train, X_val, X_test
         │
         ├─ model.fit(X_train_erro, y_train_erro)
         ├─ model.predict(X_val_erro, X_test_erro)
         │
         ├─ ŷ_final = ŷ_ARIMA + ŷ_ML_error    ← reconstrução aditiva
         └─ Salvar → data/result/<experiment_id>/<serie>_1amv1.pkl
```

### 3.3 Formulação Matemática

**Passo 1 — Modelo Linear (ARIMA):**

$$\hat{y}^{\text{ARIMA}}_t = \text{ARIMA}(p, d, q)(P, D, Q)_m$$

**Passo 2 — Série de Resíduos:**

$$e_t = y_t - \hat{y}^{\text{ARIMA}}_t$$

**Passo 3 — Modelo ML sobre os Resíduos:**

$$\hat{e}_{t+h} = f_{\text{ML}}\left(e_t, e_{t-1}, \ldots, e_{t-p}\right)$$

**Passo 4 — Previsão Final (combinação aditiva):**

$$\hat{y}^{\text{híbrido}}_{t+h} = \hat{y}^{\text{ARIMA}}_{t+h} + \hat{e}^{\text{ML}}_{t+h}$$

### 3.4 Diferença entre `arima_mlp.ipynb` e `arima_mlpens.ipynb`

| Aspecto | `arima_mlp` (simples) | `arima_mlpens` (ensemble) |
|---------|----------------------|--------------------------|
| Modelo ML | `MLPRegressor` único | `CustomBaggingRegressor` com N estimadores MLP |
| Ativação | `'identity'` | `'logistic'` |
| Busca | Grade simples de `hidden_layer_sizes` | Grade de `n_estimators × max_samples × max_features` |
| Pós-processamento | Direto (previsão única) | Requer `residual_ens_agg.exec_ensemble` para agregar estimadores |
| `model_name` | `'amv1'` | `'ammean{n}{s}{f}'` |

---

## 4. Guia Prático: Como Executar os Experimentos

### 4.1 Pré-condições

1. Verificar que os datasets estão em `data/raw/` com a coluna `y`.
2. Verificar que todos os datasets usados estão registrados em `BASE_INFORMATION` (`src/config.py`).
3. Confirmar que os datasets a executar estão listados em `BASE_NAME_LIST` (`src/config.py`).
4. Verificar o ambiente Python ativo possui as dependências instaladas:
   ```bash
   pip install -r requirements.txt
   ```

### 4.2 Ordem de Execução

```
ETAPA 1 — Modelo de referência linear
────────────────────────────────────────────────────────
[1] notebook/single_models/arima_exec.ipynb
    └─ Salva: data/result/<experiment_id>/<serie>_1arima.pkl

ETAPA 2 — Modelos únicos de ML (independentes entre si)
────────────────────────────────────────────────────────
[2a] notebook/single_models/mlp_exec.ipynb
[2b] notebook/single_models/svr_exec.ipynb
[2c] notebook/single_models/elm_exec.ipynb
[2d] notebook/single_models/scn_exec.ipynb
     ... (qualquer ordem, todos independentes)

ETAPA 3 — Sistema híbrido residual (ARIMA obrigatório antes)
────────────────────────────────────────────────────────
[3a] notebook/residual_hydridsystem/arima_mlp.ipynb
[3b] notebook/residual_hydridsystem/arima_svr.ipynb
[3c] notebook/residual_hydridsystem/arima_mlpens.ipynb  ← gera BaggingRegressor
     ... (qualquer ordem dentro da etapa 3)

ETAPA 4 — Pós-processamento de ensemble residual (após etapa 3)
────────────────────────────────────────────────────────
[4]  notebook/residual_hydridsystem/residual_ens_agg.ipynb
     └─ Requer: .pkl dos modelos ensemble (arima_mlpens, arima_svr_ens, etc.)

ETAPA 5 — Combinação entre modelos (opcional)
────────────────────────────────────────────────────────
[5]  notebook/residual_hydridsystem/error_combining.ipynb
     └─ model_comb.exec_ensemble()

ETAPA 6 — Análise de resultados
────────────────────────────────────────────────────────
[6]  notebook/calculate_metrics_v2.ipynb
     └─ metrics.open_fold_result()  →  df_mean_metrics, df_all_metrics
```

### 4.3 Parâmetros-chave para Controlar a Execução

| Parâmetro | Onde definir | Efeito |
|-----------|-------------|--------|
| `experiment_id` | Início de cada notebook | Define a subpasta de saída em `data/result/`. Mantenha consistente entre etapas. |
| `force = False` | Início de cada notebook | Pula séries já executadas (leitura do `.pkl` existente). Use `True` para re-executar. |
| `model_exec = 10` | Início de cada notebook | Número de repetições por configuração de hiperparâmetro. |
| `BASE_NAME_LIST` | `src/config.py` | Lista de séries a processar. Comente as que não deseja executar. |
| `horizon` | `experiment_params` | Deve ser **o mesmo** em todas as etapas de um mesmo experimento. |

### 4.4 Verificando Resultados Gerados

Após a execução, os arquivos `.pkl` ficam em:

```
data/result/
└── <experiment_id>/
    ├── <serie>_1arima.pkl
    ├── <serie>_1mlp.pkl
    ├── <serie>_1amv1.pkl       ← híbrido ARIMA + MLP simples
    ├── <serie>_1ammean2040.pkl ← híbrido ARIMA + MLP Bagging
    └── ...
```

Para carregar e inspecionar qualquer resultado:

```python
from model.generics import open_saved_result, format_names

fold, title = format_names('chamados', 'marecifesamu.txt', '1mlp')
resultados = open_saved_result(title)

# Cada elemento é um dict com 'experiment' e 'val_metric'
for r in resultados:
    print(r['val_metric'])
    print(r['experiment'].metrics_results['test_metrics'])
```

### 4.5 Lendo Resultados Agregados de um Experimento Completo

```python
from metrics.metrics import open_fold_result

df_mean, df_all, df_prevs = open_fold_result(
    experiment_id       = 'chamados',
    group_metrics_name  = 'val_metrics',
    metric              = 'RMSE'
)

# df_mean  → métricas médias por (série × modelo)
# df_all   → métricas de cada execução individual
# df_prevs → previsões do melhor modelo por série + valores reais
```

---

## 5. Diagrama Completo do Fluxo de Experimentos

```
config.BASE_NAME_LIST
  │
  ├─────────────────────────────────────────────────┐
  ▼                                                 ▼
arima_exec.ipynb                            mlp_exec.ipynb
  │  auto_arima (walk-forward)              svr_exec.ipynb
  │  .pkl: <serie>_1arima                   elm_exec.ipynb
  │                                         ...
  │  ŷ_ARIMA                                ŷ_ML
  │                                           │
  ▼                                           │
arima_mlp.ipynb ◄──── depende de ─────────────┘
  │  e_t = y_t - ŷ_ARIMA
  │  MLP.fit(lags de e_t)
  │  ŷ_híbrido = ŷ_ARIMA + ŷ_ML
  │  .pkl: <serie>_1amv1
  │
  ▼
arima_mlpens.ipynb
  │  BaggingRegressor(MLP) sobre e_t
  │  .pkl: <serie>_1ammeanNSF
  │
  ▼
residual_ens_agg.ipynb
  │  Agrega estimadores internos do Bagging
  │  select_type: median | cons | ds | distres | oracle
  │  .pkl: <serie>_1ammeanNSFc<select_type>...
  │
  ▼
calculate_metrics_v2.ipynb
  └─ Tabela comparativa de todas as abordagens
```
