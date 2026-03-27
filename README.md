# Experimentos em Previsão de Séries Temporais com Sistemas Híbridos

Repositório de código do mestrado em Ciência da Computação. O projeto investiga o desempenho de **sistemas híbridos residuais** (ARIMA + modelos de Machine Learning) e **ensembles com seleção dinâmica** aplicados à previsão de séries temporais univariadas de domínios reais: demanda de serviços de emergência (SAMU), acidentes de trânsito, consumo de energia elétrica e séries benchmark da literatura.

A hipótese central é que a decomposição $y_t = L_t + N_t$ — onde um modelo linear (ARIMA) captura a componente $L_t$ e um modelo não-linear (MLP, SVR, ELM, SCN) captura o resíduo $N_t$ — produz previsões superiores às de modelos isolados, especialmente quando combinada com estratégias de seleção dinâmica de estimadores.

---

## Sumário

- [Estrutura do Repositório](#estrutura-do-repositório)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Datasets](#datasets)
- [Documentação Modular](#documentação-modular)

---

## Estrutura do Repositório

```
experiments/
│
├── data/
│   └── raw/                        # Séries temporais no formato CSV (coluna única 'y')
│       ├── *.txt                   # Séries benchmark (airlines, sunspot, milk, etc.)
│       ├── samu/                   # Chamadas SAMU por município (format_samu.ipynb)
│       └── consumo_energia/        # Consumo elétrico regional ANEEL (format_data.ipynb)
│
├── docs/                           # Documentação modular gerada durante o mapeamento
│   ├── 01_fundacoes_e_dados.md
│   ├── 02_modelos_e_metricas.md
│   ├── 03_pos_processamento_e_integracoes.md
│   ├── 04_experimentos_e_hibridos.md
│   ├── 05_sistemas_hibridos_e_ensembles.md
│   └── 06_scrapping_e_utilitarios.md
│
├── notebook/
│   ├── single_models/              # Experimentos com modelos únicos (ARIMA, MLP, SVR, ELM, SCN...)
│   ├── residual_hydridsystem/      # Sistemas híbridos residuais (ARIMA + ML) e ensembles
│   ├── pert_exps/                  # Experimentos com modelo perturbativo iterativo
│   ├── perturbative_exps_deprec/   # Versões anteriores do perturbativo (depreciado)
│   ├── acf_analysis.ipynb          # Análise exploratória ACF/PACF das séries
│   ├── calculate_metrics_v2.ipynb  # Leitura e comparação de resultados salvos
│   ├── plot_test_set.ipynb         # Visualização das previsões no conjunto de teste
│   ├── recife_acidentes_format.ipynb # Formatação dos dados de acidentes de trânsito
│   ├── scrapping.ipynb             # Web scraping G1/Globo (experimental)
│   └── g1scrapping.py              # Módulo auxiliar de scraping do G1
│
├── src/
│   ├── config.py                   # Configuração global: caminhos, splits, BASE_INFORMATION
│   ├── gemini.py                   # Protótipo do algoritmo SC-III (Stochastic Config. Network)
│   ├── input/
│   │   └── input.py                # Pipeline de ingestão e pré-processamento
│   ├── metrics/
│   │   └── metrics.py              # Métricas: MSE, RMSE, MAE, MAPE, Theil-U, ARV, IA, POCID
│   ├── model/
│   │   ├── arima.py                # ARIMA via pmdarima (auto_arima + walk-forward)
│   │   ├── elm.py                  # Extreme Learning Machine
│   │   ├── scn.py                  # Stochastic Configuration Network (SC-III)
│   │   ├── lstm.py                 # LSTM (TensorFlow)
│   │   ├── kan_lbfgs.py            # Kolmogorov–Arnold Network
│   │   ├── custom_bagging.py       # BaggingRegressor customizado
│   │   ├── generics.py             # Utilitários: fit/predict, serialização, inversão de transformações
│   │   ├── single_ml_model_exp.py  # Wrapper SKlearnModel (modelos únicos)
│   │   ├── hybrid_system_exp.py    # Wrappers: Additive, RecursiveAdditive, NonLinear, ResidualCombination
│   │   ├── perturbative.py         # Wrapper Perturbative (ensemble iterativo)
│   │   ├── grid_search_exp.py      # Busca de hiperparâmetros via ParameterGrid
│   │   └── neural_forecast_exp.py  # Integração com NeuralForecast (NHITS, DeepAR, etc.)
│   └── posmodel/
│       ├── model_comb.py           # Combinação de ensemble entre modelos distintos
│       └── residual_ens_agg.py     # Agregação dos estimadores internos do BaggingRegressor
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Instalação

### Pré-requisitos

- Python >= 3.7
- `pip`

### Passo a passo

**1. Clonar o repositório**

```bash
git clone https://github.com/domingos108/experiments.git
cd experiments
```

**2. Criar e ativar o ambiente virtual**

```bash
# Criar o venv
python -m venv .venv

# Ativar (Linux/macOS)
source .venv/bin/activate

# Ativar (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

**3. Instalar as dependências**

```bash
pip install -r requirements.txt
```

**4. Instalar o pacote local** (necessário para os imports `from model import ...`, `from input import ...`, etc.)

```bash
pip install -e .
```

**5. Verificar a instalação**

```bash
python -c "from src.input import input; print('OK')"
```

### Dependências principais

| Pacote | Uso |
|--------|-----|
| `pandas == 2.2.1` | Manipulação de séries e DataFrames |
| `scikit-learn` | Modelos ML (MLP, SVR, BaggingRegressor, etc.) |
| `pmdarima` | ARIMA automático e testes de estacionariedade |
| `sktime` | Teste KPSS de estacionariedade |
| `neuralforecast` | Modelos neurais de previsão (NHITS, DeepAR, Informer) |
| `tensorflow` | LSTM e redes neurais customizadas |
| `matplotlib` | Visualizações |
| `jupyterlab` | Execução dos notebooks |

---

## Como Usar

A execução segue uma **ordem lógica de dependências**. Respeite a sequência abaixo para garantir que os arquivos `.pkl` necessários existam antes de cada etapa.

### Etapa 0 — Configuração

Edite `src/config.py`:

1. Verifique `BASE_NAME_LIST` — lista de séries a processar.
2. Confirme que cada série está em `BASE_INFORMATION` com `freq`, `m` e `lag_size`.
3. Ajuste `TEST_SIZE` e `VAL_SIZE` se necessário (padrão: `0.1` cada).

### Etapa 1 — Análise Exploratória (EDA)

```
notebook/acf_analysis.ipynb
```

Visualiza ACF/PACF, testa estacionariedade (KPSS) e determina o `lag_size` adequado para cada série.

### Etapa 2 — Modelo de Referência Linear

```
notebook/single_models/arima_exec.ipynb
```

**Obrigatório antes de qualquer sistema híbrido.** Salva as previsões ARIMA em `data/result/<experiment_id>/<serie>_1arima.pkl`.

### Etapa 3 — Modelos Únicos de ML

```
notebook/single_models/mlp_exec.ipynb
notebook/single_models/svr_exec.ipynb
notebook/single_models/elm_exec.ipynb
notebook/single_models/scn_exec.ipynb
...
```

Todos independentes entre si. Cada notebook realiza busca de hiperparâmetros via grid search com `model_exec=10` repetições.

### Etapa 4 — Sistemas Híbridos Residuais

```
notebook/residual_hydridsystem/arima_mlp.ipynb      # Aditivo simples
notebook/residual_hydridsystem/arima_mlpens.ipynb   # Aditivo com Bagging
notebook/residual_hydridsystem/arima_svr.ipynb
notebook/residual_hydridsystem/arima_scn.ipynb
...
```

Requerem o `.pkl` do ARIMA (Etapa 2). O modelo ML aprende a série de resíduos $e_t = y_t - \hat{y}^{\text{ARIMA}}_t$.

### Etapa 5 — Pós-processamento de Ensemble

```
notebook/residual_hydridsystem/residual_ens_agg.ipynb
```

Agrega os estimadores internos do BaggingRegressor usando estratégias de seleção dinâmica (`ds`, `median`, `cons`, `distres`, `oracle`). Requer os `.pkl` dos modelos ensemble (Etapa 4).

### Etapa 6 — Análise de Resultados

```
notebook/calculate_metrics_v2.ipynb
notebook/plot_test_set.ipynb
```

Lê todos os `.pkl` de um `experiment_id`, gera tabelas comparativas e visualizações das previsões no conjunto de teste.

### Configurando um Experimento

Em cada notebook, os parâmetros principais são:

```python
experiment_id = 'nome_do_experimento'  # subpasta em data/result/
model_exec    = 10                     # repetições por config de hiperparâmetro
force         = False                  # True = re-executar mesmo se .pkl existir
```

Os resultados são salvos automaticamente em:

```
data/result/<experiment_id>/<serie>_<horizon><model_name>.pkl
```

---

## Datasets

Todos os datasets estão em `data/raw/` como arquivos `.txt` com uma única coluna `y`.

| Grupo | Arquivos | Frequência | Fonte |
|-------|----------|------------|-------|
| SAMU (Recife) | `marecifesamu.txt`, `majaboataosamu.txt`, `maolindasamu.txt`, `mapaulistasamu.txt` | Diária | Dados abertos SAMU |
| Acidentes (Recife) | `recifeaccday.txt`, `recifeaccweek.txt`, `recifeaccmonth.txt`, `marecacc.txt` | Diária/Semanal/Mensal | Prefeitura do Recife |
| Energia Elétrica | `consumo[co/ne/no/sd/sul]formated.txt` | Mensal | ANEEL/EPE |
| Benchmark | `airlines.txt`, `sunspot.txt`, `milk.txt`, `woolyrnq.txt`, `temperature.txt`, etc. | Variada | Literatura (M-competition, R datasets) |

Para adicionar uma nova série, consulte [docs/01_fundacoes_e_dados.md](docs/01_fundacoes_e_dados.md) — Seção 5.

---

## Documentação Modular

| Documento | Conteúdo |
|-----------|----------|
| [01 — Fundações e Dados](docs/01_fundacoes_e_dados.md) | Configuração global, formato dos dados, pipeline de pré-processamento |
| [02 — Modelos e Métricas](docs/02_modelos_e_metricas.md) | Métricas de avaliação (MSE, RMSE, MAPE, IA, POCID...), wrappers de experimento, busca de hiperparâmetros |
| [03 — Pós-processamento e Integrações](docs/03_pos_processamento_e_integracoes.md) | Combinação de ensemble, seleção dinâmica, algoritmo SC-III |
| [04 — Experimentos e Híbridos](docs/04_experimentos_e_hibridos.md) | Guia prático de execução, ordem dos notebooks, diferença entre modelo único e híbrido |
| [05 — Sistemas Híbridos e Ensembles](docs/05_sistemas_hibridos_e_ensembles.md) | Arquiteturas híbridas detalhadas (Additive, RecursiveAdditive, NonLinear, Perturbative), formulação matemática |
| [06 — Scraping e Utilitários](docs/06_scrapping_e_utilitarios.md) | Extração e formatação de dados externos, guia de atualização das bases |
