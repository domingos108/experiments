# 03 — Pós-processamento e Integrações

Este documento descreve o que acontece com as previsões após os modelos serem treinados: as estratégias de combinação de ensemble, a seleção dinâmica de modelos e o papel do script `gemini.py` dentro do projeto.

---

## 1. Visão Geral da Etapa de Pós-processamento

O pós-processamento atua sobre previsões **já geradas e salvas** (em `.pkl`). Ele não re-treina modelos — em vez disso, combina, seleciona ou agrega as saídas de múltiplos modelos para produzir uma previsão final mais robusta.

Existem dois módulos principais nesta etapa:

| Arquivo | Responsabilidade |
|---------|-----------------|
| `posmodel/model_comb.py` | Combinação de ensemble entre modelos independentes (média, mediana, seleção dinâmica). |
| `posmodel/residual_ens_agg.py` | Agregação dos estimadores internos de um BaggingRegressor para o sistema híbrido ARIMA + ML. |

---

## 2. Combinação de Modelos (`posmodel/model_comb.py`)

### 2.1 Estratégias de Combinação (`exec_ensemble`)

Ponto de entrada principal para combinar as previsões de uma **lista de modelos distintos** sobre uma mesma série.

```
Carregar previsões salvas de cada modelo (.pkl)
    │
    ▼
Pivotar: DataFrame [index × model_name → prev]
    │
    ▼
Aplicar estratégia de combinação
    │
    ▼
metrics.format_metrics_results()
    │
    ▼
Salvar resultado combinado em .pkl
```

| `ens_type` | Descrição |
|------------|-----------|
| `'mean'`   | Média simples das previsões de todos os modelos. |
| `'median'` | Mediana das previsões de todos os modelos. |
| `'ds'`     | Seleção dinâmica baseada em distância euclidiana (`dinanic_selection`). |

### 2.2 Seleção Dinâmica por Distância Euclidiana (`dinanic_selection`)

Estratégia de seleção de subconjunto de modelos em tempo de teste, sem re-treinamento:

1. Para cada ponto do conjunto de teste, busca as $r_c$ janelas mais similares na série de validação usando **distância euclidiana** sobre o vetor de lags.
2. Para cada janela similar, calcula o erro quadrático de cada modelo.
3. Seleciona os $k$ modelos com menor erro médio nas janelas similares.
4. A previsão final é a **média** das $k$ previsões selecionadas.

**Parâmetros (`ds_args`):**

| Parâmetro  | Descrição |
|------------|-----------|
| `rc`       | Número de vizinhos mais próximos usados para avaliar o desempenho histórico. |
| `k`        | Número de modelos selecionados para combinar. |
| `lag_size` | Tamanho da janela de lags para o cálculo da distância. |

### 2.3 Variantes de Seleção Dinâmica

| Função | Critério de similaridade |
|--------|--------------------------|
| `dinanic_selection` | Janelas mais similares na validação (distância euclidiana). |
| `most_recent_dinanic_selection` | Erro do passo imediatamente anterior (`shift(1)`). |
| `seazonal_dinanic_selection` | Erro do mesmo ponto na sazonalidade anterior (`shift(12)`). |
| `oracle` | Seleciona sempre o modelo de menor erro real — **limite inferior teórico**, usado como referência. |

### 2.4 Combinação Linear Supervisionada (`exec_linear_comb`)

Treina um modelo sklearn (ex: Ridge, SVR) para aprender os **pesos ótimos** de combinação das componentes perturbativas, usando as colunas de `df_pert` como features e o valor real como target.

---

## 3. Agregação de Ensemble Residual (`posmodel/residual_ens_agg.py`)

Este módulo é a camada de pós-processamento específica para o sistema **ARIMA + BaggingRegressor**, onde cada estimador interno do Bagging recebeu a série de resíduos do ARIMA.

### 3.1 Fluxo Principal (`exec_ensemble`)

```
Carregar previsão do ARIMA (ts_forecast) e série de resíduos (error_series)
    │
    ▼
predict_estimators()
  └─ Para cada estimador base no BaggingRegressor:
       ├─ extrair features de bagging (colunas amostradas)
       ├─ model.predict(X)
       └─ inverter MinMaxScaler → erro no espaço original
    │
    ▼
DataFrame de previsões individuais [N_amostras × N_estimadores]
    │
    ▼
Aplicar select_type
    │
    ▼
ŷ_final = ŷ_ARIMA + ŷ_erro_selecionado
    │
    ▼
metrics.format_metrics_results()  →  Salvar .pkl
```

### 3.2 Estratégias de Seleção (`select_type`)

| `select_type`   | Descrição |
|-----------------|-----------|
| `'median'`      | Mediana das previsões de todos os estimadores. |
| `'cons'`        | Consenso ponderado: usa quantis Q75/Q25 dependendo da maioria de sinais positivos/negativos (`get_conssesuos`). |
| `'ds'`          | Seleção dinâmica por distância euclidiana na janela de validação. |
| `'dsseazonal'`  | Seleção dinâmica por sazonalidade (`shift(12)`). |
| `'mostrecent'`  | Seleção dinâmica pelo erro mais recente (`shift(1)`). |
| `'distres'`     | Seleção por **divergência de resíduos**: escolhe os $k$ modelos com maior desacordo direcional entre si (diversidade de ensemble). |
| `'oracle'`      | Seleção ótima a posteriori — cota inferior de erro, apenas para benchmarking. |

### 3.3 Reconstrução da Previsão Final

Para todos os `select_type` que operam sobre o erro (todos exceto `'ds'`, `'dsseazonal'`, `'mostrecent'` e `'oracle'`):

$$\hat{y}^{\text{final}}_t = \hat{y}^{\text{ARIMA}}_t + \hat{e}^{\text{selecionado}}_t$$

As estratégias `'ds'`, `'dsseazonal'`, `'mostrecent'` e `'oracle'` já operam sobre a soma $\hat{y}^{\text{ARIMA}} + \hat{e}_i$ antes de selecionar, retornando diretamente a previsão combinada.

> **Nota:** Não há desnormalização explícita nesta camada — ela é responsabilidade de `predict_estimators`, que chama `min_max_scaler.inverse_transform` imediatamente após cada estimador.

---

## 4. O Script `gemini.py` — Algoritmo SC-III

### 4.1 O que é

`gemini.py` implementa o **Algoritmo SC-III (Stochastic Configuration Network — variante III)**, uma Rede Neural de Camada Única (SLFN) com pesos de entrada **aleatórios** e pesos de saída calculados analiticamente pela **pseudo-inversa**.

O arquivo é uma **prova de conceito / protótipo**, provavelmente gerado com auxílio de IA generativa (daí o nome `gemini.py`). A versão de produção utilizada nos experimentos está em `src/model/scn.py`.

### 4.2 Funcionamento do Algoritmo

```
Dados de entrada X (N×d) e target T (N×m)
    │
    ▼
e_0 = T  (erro inicial = target)
    │
    ▼
Para L = 1, 2, ..., L_max  enquanto  ||e_{L-1}||_F > ε:
    │
    ├─ Para t = 1, ..., T_max (tentativas aleatórias):
    │     ├─ Sortear ω_L ~ U(-λ, λ)  e  b_L ~ U(-λ, λ)
    │     ├─ Calcular h_L(X) = sigmoid(X·ω_L + b_L)
    │     └─ Calcular ξ_{L} = Σ_q  (e_{L-1,q}ᵀ h_L)² / (h_Lᵀ h_L)
    │
    ├─ Selecionar (ω*, b*) que maximiza ξ_L
    ├─ Expandir H_L = [H_{L-1} | h*]
    ├─ β* = H_L⁺ · T  (mínimos quadrados via pseudo-inversa)
    └─ e_L = H_L β* - T  (resíduo atualizado)
    │
    ▼
Previsão: ŷ = H_test · β*
```

### 4.3 Parâmetros da Classe `SC_III_Algorithm`

| Parâmetro         | Descrição |
|-------------------|-----------|
| `L_max`           | Número máximo de neurônios ocultos. |
| `epsilon`         | Tolerância da norma de Frobenius do erro $\|e_L\|_F$. |
| `T_max`           | Número de configurações aleatórias tentadas por neurônio. |
| `gamma_min/max`   | Intervalo para geração do conjunto $\Gamma$ (reservado para variantes do SC). |
| `num_gamma_steps` | Número de valores de $\gamma$ no linspace. |
| `lambda_limit`    | Limite $\lambda$ para sorteio uniforme de $\omega$ e $b$: $U(-\lambda, \lambda)$. |

### 4.4 Métrica de Importância $\xi$

$$\xi_{L,q} = \frac{\left( e_{L-1,q}^\top h_L(X) \right)^2}{h_L(X)^\top h_L(X)}$$

Mede a projeção (correlação quadrática normalizada) do resíduo atual sobre a ativação do neurônio candidato. O neurônio que maximiza $\sum_q \xi_{L,q}$ é selecionado.

---

## 5. Passo a Passo Teórico para Testar as Funções

### 5.1 Testar `exec_ensemble` em `model_comb.py`

```python
from posmodel.model_comb import exec_ensemble

# Pré-requisito: ter resultados de ao menos 2 modelos salvos em .pkl
# para a mesma série e experiment_id.

exec_ensemble(
    experiment_id = 'meu_experimento',
    modelo_list   = ['mlp', 'svr'],       # nomes dos modelos já treinados
    ens_type      = 'mean',               # 'mean' | 'median' | 'ds'
    ds_args       = {},                   # necessário apenas para ens_type='ds'
    horizon       = 1
)
# Resultado salvo em: data/result/meu_experimento/<serie>_combmean.pkl
```

### 5.2 Testar `exec_ensemble` em `residual_ens_agg.py`

```python
from posmodel.residual_ens_agg import exec_ensemble

experiment_params = {
    'select_type': 'median',   # ou 'cons', 'ds', 'distres', 'oracle' ...
    'linear_model_name': 'arima',
    'test_size':  0.1,
    'val_size':   0.1,
    'horizon':    1,
    'lag_size':   7,
    'diff_kpss':  False,
    'ds_args': {'rc': 0.5, 'k': 3, 'lag_size': 7}  # necessário para select_type='ds'
}

exec_ensemble(
    experiment_params = experiment_params,
    experiment_id     = 'meu_experimento',
    base_name         = 'marecifesamu.txt',
    model_name        = '1arima_mlp'      # nome do .pkl do sistema híbrido já treinado
)
```

### 5.3 Testar o Algoritmo SC-III (`gemini.py`)

```python
import numpy as np
from src.gemini import SC_III_Algorithm

# 1. Gerar dados de exemplo
X_train = np.linspace(-3, 3, 200).reshape(-1, 1)
y_train = (X_train**2 + np.random.normal(0, 0.3, X_train.shape)).reshape(-1, 1)

# 2. Instanciar o modelo
modelo = SC_III_Algorithm(
    L_max           = 30,    # máximo de neurônios
    epsilon         = 0.3,   # tolerância de erro
    T_max           = 100,   # tentativas por neurônio
    gamma_min       = 0.1,
    gamma_max       = 1.0,
    num_gamma_steps = 10,
    lambda_limit    = 1.0
)

# 3. Treinar
modelo.fit(X_train, y_train)
# Acompanhar: número de neurônios selecionados e norma do erro por iteração

# 4. Avaliar
X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
y_pred = modelo.predict(X_test)

# 5. Verificar sanidade
print(f'Neurônios selecionados: {len(modelo.omega_star)}')
print(f'Shape da predição: {y_pred.shape}')  # esperado: (100, 1)
```

### 5.4 Verificar Resultados Salvos

```python
from model.generics import open_saved_result, format_names

fold, title = format_names('meu_experimento', 'marecifesamu.txt', 'combmean')
resultados = open_saved_result(title)

for r in resultados:
    print(r['experiment'].metrics_results['test_metrics'])
```

---

## 6. Relação entre os Módulos de Pós-processamento

```
Modelos treinados (.pkl)
  ├─ SKlearnModel / Additive / Perturbative
  │
  ▼
posmodel/model_comb.py          posmodel/residual_ens_agg.py
  ├─ mean / median                 ├─ median / cons
  ├─ dinanic_selection             ├─ ds / dsseazonal / mostrecent
  ├─ exec_linear_comb              ├─ distres
  └─ oracle (benchmark)            └─ oracle (benchmark)
  │                                │
  └────────────┬───────────────────┘
               ▼
    metrics.format_metrics_results()
               │
               ▼
    Salvar .pkl  →  metrics.open_fold_result()  →  análise comparativa
```
