# 05 — Sistemas Híbridos e Ensembles

Este documento descreve a lógica matemática e computacional por trás das arquiteturas híbridas do projeto, que constituem a contribuição principal do mestrado, além de um guia prático para executar os notebooks avançados.

---

## 1. Motivação: Por que Hibridizar?

Séries temporais reais apresentam componentes lineares (tendência, sazonalidade) e não-lineares (dinâmicas complexas, efeitos extremos). Modelos puros tendem a capturar apenas um dos aspectos:

- **ARIMA**: excelente para estrutura linear/sazonal; falha em padrões não-lineares.
- **ML (MLP, SVR, ELM, SCN)**: captura não-linearidades; pode ignorar estrutura temporal global.

A **hibridização residual** combina os dois mundos: o modelo linear prevê o que consegue, e o modelo não-linear aprende apenas o que sobrou.

---

## 2. Taxonomia das Arquiteturas Híbridas

O projeto implementa cinco arquiteturas distintas em `src/model/hybrid_system_exp.py`:

| Classe | Arquivo/Notebook | Estratégia |
|--------|-----------------|------------|
| `Additive` | `arima_mlp.ipynb` | ARIMA + ML(resíduo) — 1 passo |
| `RecursiveAdditive` | `arima_mlpnonlinear.ipynb` | ARIMA + ML(resíduo) iterativo com `learning_rate` |
| `NonLinear` | `arima_mlpnonlinear.ipynb` | ML com entrada configurável: série, erro, previsão linear |
| `ResidualCombination` | `error_combining.ipynb` | Meta-aprendiz treinado sobre [ŷ_ARIMA, ŷ_ML] |
| `Perturbative` | `pert_exps/pert_mlp.ipynb` | Ensemble iterativo perturbativo sobre a série original |

---

## 3. Arquitetura 1 — Aditiva Simples (`Additive`)

### 3.1 Formulação Matemática

**Passo 1 — ARIMA modela a componente linear:**

$$\hat{y}^{\text{ARIMA}}_t = \text{ARIMA}(p, d, q)(P, D, Q)_m$$

**Passo 2 — Série de resíduos:**

$$e_t = y_t - \hat{y}^{\text{ARIMA}}_t$$

**Passo 3 — ML modela a componente não-linear residual:**

$$\hat{e}_{t+h} = f_{\text{ML}}\!\left(e_t, e_{t-1}, \ldots, e_{t-p}\right)$$

**Passo 4 — Previsão final (combinação aditiva):**

$$\hat{y}^{\text{híbrido}}_{t+h} = \hat{y}^{\text{ARIMA}}_{t+h} + \hat{e}^{\text{ML}}_{t+h}$$

### 3.2 Hipótese Central

$$y_t = L_t + N_t$$

Onde $L_t$ é a componente linear capturada pelo ARIMA e $N_t$ é a componente não-linear (resíduo). O ML deve aprender $N_t$ a partir de seus próprios lags.

### 3.3 Notebooks

| Notebook | Modelo ML | Observações |
|----------|-----------|-------------|
| `arima_mlp.ipynb` | `MLPRegressor(activation='identity')` | MLP simples sobre os resíduos |
| `arima_svr.ipynb` | `SVR` | SVM de regressão sobre os resíduos |
| `arima_elm.ipynb` | `ELM` | Extreme Learning Machine nos resíduos |
| `arima_scn.ipynb` | `SCN_III` | Stochastic Configuration Network nos resíduos |

---

## 4. Arquitetura 2 — Aditiva com Ensemble de Bagging (`Additive` + `BaggingRegressor`)

### 4.1 Formulação

Mesma base aditiva, mas o modelo ML é um **BaggingRegressor** com $N$ estimadores base:

$$\hat{e}_{t+h} = \text{Bagging}\!\left\{ f^{(1)}_{\text{ML}}, f^{(2)}_{\text{ML}}, \ldots, f^{(N)}_{\text{ML}} \right\}$$

Cada estimador $f^{(i)}$ é treinado em um subconjunto amostrado com reposição (`max_samples`) e subconjunto de features (`max_features`).

### 4.2 Grades de Hiperparâmetros (exemplo de `arima_mlpens.ipynb`)

```python
for n_estimators in [20, 40, 80]:          # número de estimadores
    for max_samples in [0.4, 0.6, 0.8]:    # fração de amostras por estimador
        for max_features in [0.4, 0.6, 0.8]:  # fração de features por estimador
```

### 4.3 Pós-processamento Obrigatório

O Bagging gera múltiplas previsões individuais. Após o treinamento, é preciso executar `residual_ens_agg.exec_ensemble()` para agregar os estimadores:

```
arima_mlpens.ipynb
  └─ Salva: <serie>_1ammeanNSF.pkl  (N estimadores, S amostras, F features)
         │
         ▼
residual_ens_agg.ipynb
  └─ exec_ensemble(select_type='ds' | 'median' | 'cons' | ...)
  └─ Salva: <serie>_1ammeanNSFcdsrc…k….pkl
```

---

## 5. Arquitetura 3 — Aditiva Recursiva (`RecursiveAdditive`)

### 5.1 Formulação

O modelo ML é aplicado **iterativamente** sobre o resíduo acumulado:

$$\hat{y}^{(0)}_t = \hat{y}^{\text{ARIMA}}_t$$

$$e^{(i)}_t = y_t - \hat{y}^{(i-1)}_t$$

$$\hat{y}^{(i)}_t = \hat{y}^{(i-1)}_t + \lambda \cdot f^{(i)}_{\text{ML}}\!\left(e^{(i)}_t, \ldots\right)$$

Após `max_it` iterações:

$$\hat{y}^{\text{final}}_t = \hat{y}^{\text{ARIMA}}_t + \lambda \sum_{i=1}^{T} \hat{e}^{(i)}_t$$

O parâmetro `learning_rate` $(\lambda)$ controla a contribuição de cada correção, evitando overfitting cumulativo.

---

## 6. Arquitetura 4 — Entrada Configurável (`NonLinear`)

### 6.1 Formulação

Ao contrário do `Additive`, o `NonLinear` permite controlar o que entra no modelo ML via flags em `experiment_params`:

| Flag | Entrada adicionada ao vetor de features |
|------|----------------------------------------|
| `use_linear=True` | $\hat{y}^{\text{ARIMA}}_t$ (previsão linear atual) |
| `use_error=True` | $e_{t}, e_{t-1}, \ldots, e_{t-p}$ (lags do resíduo) |
| `use_series=True` | $y_{t}, y_{t-1}, \ldots, y_{t-p}$ (lags da série original) |

O alvo é sempre $y_{t+h}$ (série original), e a inversão de normalização é feita com escaler separado para $X$ e $y$.

---

## 7. Arquitetura 5 — Combinação Residual Supervisionada (`ResidualCombination`)

### 7.1 Formulação

Treina um **meta-aprendiz** para aprender os pesos ótimos de combinação entre a componente linear e não-linear:

$$\hat{y}^{\text{meta}}_{t} = g\!\left(\hat{y}^{\text{ARIMA}}_t,\; \hat{e}^{\text{ML}}_t,\; \hat{y}^{\text{ARIMA}}_t + \hat{e}^{\text{ML}}_t\right)$$

As features de entrada do meta-aprendiz são:

```python
df_input = {
    'linear_forecast':    ŷ_ARIMA,
    'nonlinear_forecast': ŷ_ML_error,
    'ens':                ŷ_ARIMA + ŷ_ML_error  # combinação simples
}
```

O meta-aprendiz (`PoissonRegressor`, `Ridge`, etc.) é treinado no treino e avaliado no teste.

---

## 8. Arquitetura 6 — Perturbativa (`Perturbative`)

### 8.1 Formulação

Sistema de ensemble **iterativo e auto-corretivo** sobre a série original:

**Iteração 0** — modelo base:
$$p_0 = f^{(0)}_{\text{ML}}(y_t, y_{t-1}, \ldots)$$

**Iteração $k$** — perturbação sobre o resíduo acumulado:
$$e^{(k)}_t = y_t - \sum_{j=0}^{k-1} \lambda_j \cdot p_j(t)$$
$$p_k = f^{(k)}_{\text{ML}}\!\left(e^{(k)}_t, e^{(k)}_{t-1}, \ldots\right)$$

**Previsão final:**
$$\hat{y}_t = \sum_{k=0}^{K} \lambda_k \cdot p_k(t)$$

### 8.2 Parâmetros de Controle

| Parâmetro | Descrição |
|-----------|-----------|
| `models_before` | Lista de `.pkl` de modelos já treinados — define $p_0, \ldots, p_{K-1}$ |
| `qtd_perturbations` | Alternativa a `models_before`: número fixo de iterações |
| `learning_rate_init` | $\lambda$ — peso de cada componente perturbativa |
| `earlystop` | Para se o erro não diminui por $N$ iterações consecutivas |
| `bagging_pct` | Se definido, cada $p_k$ é treinado em subconjunto amostrado da série |

### 8.3 Estratégia Sequencial de Execução (`pert_mlp.ipynb`)

```python
# Passo 1: treinar modelo base (mlp)
model_name_list = ['1mlpp1', '1mlpp2', '1mlpp3']
models_before_list = [
    ['1mlp'],                        # p1 usa apenas p0
    ['1mlp', '1mlpp1'],             # p2 usa p0 e p1
    ['1mlp', '1mlpp1', '1mlpp2']    # p3 usa p0, p1 e p2
]
```

Cada perturbação carrega os parâmetros dos modelos anteriores via `.pkl`, garantindo consistência entre as iterações.

---

## 9. Estratégias de Combinação do Ensemble Residual

Após gerar as previsões de todos os estimadores do Bagging, `residual_ens_agg.exec_ensemble()` aplica uma das estratégias abaixo:

### 9.1 Estratégias Estáticas

| `select_type` | Fórmula | Quando usar |
|---------------|---------|-------------|
| `'median'` | $\hat{y} = \text{mediana}(\hat{e}^{(1)}, \ldots, \hat{e}^{(N)}) + \hat{y}^{\text{ARIMA}}$ | Robusto a outliers entre estimadores |
| `'cons'` | Quantil 75% ou 25% da maioria direcional | Quando há consenso claro de sinal dos resíduos |

### 9.2 Seleção Dinâmica por Similaridade Temporal

Para cada ponto de teste $t$, seleciona os $k$ melhores estimadores com base no desempenho histórico em janelas similares:

$$\hat{y}_t = \frac{1}{k} \sum_{i \in \mathcal{S}_t} \left(\hat{y}^{\text{ARIMA}}_t + \hat{e}^{(i)}_t\right)$$

Onde $\mathcal{S}_t$ é o subconjunto de $k$ modelos com menor erro nas $r_c$ janelas de validação mais próximas (por distância euclidiana dos lags).

| `select_type` | Critério de similaridade |
|---------------|--------------------------|
| `'ds'` | $r_c$ janelas mais similares na validação (distância euclidiana) |
| `'dsseazonal'` | Mesmo ponto na sazonalidade anterior (`shift(12)`) |
| `'mostrecent'` | Erro do passo imediatamente anterior (`shift(1)`) |

### 9.3 Diversidade de Ensemble

| `select_type` | Estratégia |
|---------------|-----------|
| `'distres'` | Seleciona os $k$ modelos com **maior desacordo direcional** entre si na validação (maximiza diversidade) |
| `'oracle'` | Seleciona sempre o modelo de menor erro real — **limite inferior teórico** (benchmark) |

---

## 10. Guia Prático: Executando os Sistemas Híbridos

### 10.1 Pré-requisito: ARIMA treinado

```
notebook/single_models/arima_exec.ipynb  →  obrigatório antes de qualquer híbrido
```

### 10.2 Ordem de Execução — Sistema Aditivo Simples

```
[1] arima_exec.ipynb
    └─ data/result/<id>/<serie>_1arima.pkl

[2] residual_hydridsystem/arima_mlp.ipynb
    ├─ experiment_params['linear_model_name'] = '1arima'
    └─ data/result/<id>/<serie>_1amv1.pkl
```

### 10.3 Ordem de Execução — Ensemble Residual Completo

```
[1] arima_exec.ipynb
    └─ <serie>_1arima.pkl

[2] residual_hydridsystem/arima_mlpens.ipynb
    └─ <serie>_1ammean<N><S><F>.pkl    (BaggingRegressor treinado)

[3] residual_hydridsystem/residual_ens_agg.ipynb
    ├─ select_type: 'ds' | 'median' | 'cons' | 'distres' | 'oracle'
    ├─ grade: rc ∈ [0.1, 0.5, 0.8], k ∈ [1, 5, 10, 20, 40, 80]
    └─ <serie>_1ammean<N><S><F>c<type>rc<rc>k<k>.pkl
```

### 10.4 Ordem de Execução — Perturbativo

```
[1] single_models/mlp_exec.ipynb          →  p0 base
    └─ <serie>_1mlp.pkl

[2] pert_exps/pert_mlp.ipynb (iteração 1) →  p1
    ├─ models_before = ['1mlp']
    └─ <serie>_1mlpp1.pkl

[3] pert_exps/pert_mlp.ipynb (iteração 2) →  p2
    ├─ models_before = ['1mlp', '1mlpp1']
    └─ <serie>_1mlpp2.pkl

[4] pert_exps/pert_mlp.ipynb (iteração 3) →  p3
    ├─ models_before = ['1mlp', '1mlpp1', '1mlpp2']
    └─ <serie>_1mlpp3.pkl
```

### 10.5 Ordem de Execução — Combinação Residual Supervisionada

```
[1] arima_exec.ipynb
[2] residual_hydridsystem/arima_mlp.ipynb   →  gera 'nonlinear_forecast' no .pkl
[3] error_combining.ipynb
    ├─ experiment_params['nonlinear_model'] = '1arimamlp'
    └─ meta-aprendiz treinado sobre [ŷ_ARIMA, ŷ_ML, ŷ_ARIMA + ŷ_ML]
```

---

## 11. Parâmetros-Chave por Arquitetura

### `Additive` / `arima_mlp.ipynb`

```python
experiment_params = {
    'linear_model_name': '1arima',  # nome do .pkl do modelo linear
    'diff_kpss':         False,     # diferenciação já feita pelo ARIMA
    'horizon':           1
}
```

### `Additive` com Bagging / `arima_mlpens.ipynb`

```python
model = BaggingRegressor(n_jobs=-1)
model_parameters = {
    'estimator':    [MLPRegressor(...)],
    'n_estimators': [20, 40, 80],
    'max_samples':  [0.4, 0.6, 0.8],
    'max_features': [0.4, 0.6, 0.8]
}
```

### `Perturbative` / `pert_mlp.ipynb`

```python
experiment_params = {
    'models_before':      ['1mlp', '1mlpp1'],  # .pkl dos modelos anteriores
    'model_pertub':       MLPRegressor(...),    # modelo base de cada perturbação
    'learning_rate_init': 1,                   # λ (peso de cada componente)
    'qtd_perturbations':  None,                # None = usa models_before
    'earlystop':          None,                # ou int: parar se sem melhora
    'bagging_pct':        None                 # ou float: subconjunto de treino
}
```

### `residual_ens_agg.exec_ensemble()`

```python
experiment_params = {
    'linear_model_name': '1arima',
    'select_type':       'ds',       # estratégia de seleção/combinação
    'lag_size':          7,          # tamanho da janela de lags para distância
    'ds_args': {
        'rc': 0.5,   # fração da validação como vizinhos (ou int absoluto)
        'k':  5      # número de estimadores selecionados
    }
}
```

---

## 12. Diagrama Completo das Arquiteturas Híbridas

```
y_t  ─────────────────────────────────────────────────────────────┐
  │                                                               │
  ▼                                                               │
ARIMA(p,d,q)(P,D,Q)_m                                            │
  │  ŷ_ARIMA                                                      │
  │                                                               │
  ├──► e_t = y_t - ŷ_ARIMA  ────────────────────────────────┐    │
  │           │                                              │    │
  │    ┌──────┴──────────────────────────┐                  │    │
  │    │                                 │                  │    │
  │    ▼                                 ▼                  │    │
  │  ML_simples(lags de e_t)     BaggingRegressor           │    │
  │    │  ê_ML                   [ML_1, ..., ML_N]          │    │
  │    │                              │                     │    │
  │    │                    residual_ens_agg                │    │
  │    │                    select_type:                    │    │
  │    │                    median|cons|ds|distres|oracle   │    │
  │    │                         │  ê_ensemble              │    │
  │    │                         │                          │    │
  │  ŷ_final = ŷ_ARIMA + ê_ML   │                          │    │
  │    │                         │                          │    │
  │    └─────────────────────────┘                          │    │
  │                  │                                      │    │
  │         ResidualCombination                             │    │
  │         meta-aprendiz(ŷ_ARIMA, ê_ML, soma)             │    │
  │                  │                                      │    │
  │                  ▼                                      │    │
  │           ŷ_meta_final                                  │    │
  │                                                         │    │
  └─────────────────────────────────────────────────────────┘    │
                     │                                            │
          Perturbative (sobre y_t diretamente) ◄─────────────────┘
          p_0 + λ·p_1 + λ·p_2 + ...
```
