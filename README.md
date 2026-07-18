# Passos para instalação
- apenas para linux, caso tenha disponivel apenas linux criar o WSL
- clone do projeto
- entrar na pasta do projeto
- instala o python 3.10
- cria o ambiente: python -m venv .venv
- ativar o ambiente: source .venv/bin/activate
- pip install -e .
- pip install -r requirements.txt

# Objetivo do projeto
- motor para experimentação de modelos de séries temporais
- Exemplos de Execução de modelos implementadas na pasta notebooks, exemplos:
    - notebook/single_models/arima_exec.ipynb
    - notebook/single_models/mlp_exec.ipynb
    - notebook/single_models/svr_exec.ipynb
    - notebook/single_models/kan_exec.ipynb
    - notebook/residual_hybridsystem/arima_mlp.ipynb: Sistema híbrido residual entre arima e mlp
- Arquivos importantes:
    - data/raw: pasta que contém dados brutos e resultados
    - data/results: quardará resultados das execuções dos notebooks, criando uma pasta pelo experimental_id
    - config.py: responsável pelaa definição dos splits de treinamento, valdiação e teste, definição da lista de series temporais que serão executadas, e configurações gerais da série temporal

# Estrutura do repositório

## Diretórios principais
- `data/`
  - `raw/`
    - arquivos de séries temporais e dados brutos em que o target tem a coluna chamada de y, como `airlines.txt`, `temperature.txt`, `sunspot.txt`
  - `result/`
    - resultados e saídas organizados por experimento ou tema baseado no experimental_id
- `notebook/`
  - notebooks Jupyter de análise e experimentos, principais:
    - `calculate_metrics.ipynb`: Calcula métricas do experimental_id
    - `ensemble.ipynb`: Gera reultados de ensemble
    - `plot_test_set.ipynb`: Gera gráficos de no teste do experimental_id
  - subdiretórios:
    - `residual_hydridsystem/`
      - notebooks  para execução  de sistemas híbridos de resíduos
    - `single_models/`
      - notebooks para execução de modelos individuais

- `src/`
  - código-fonte Python
  - submódulos:
    - `config.py`: Python que orquesta a execução, tais como séries que serão executadas, porcentagem de treinamento, validação e teste, e configuração de lags e sazonalidade das séries disponíveis na pasta data/raw
    - `input/`: funções para importação e preprocessamentos, normalização e organização dos dados
    - `metrics/`: geração de métricas dos modelos
    - `model/`: implementação do funcionamento do grid search, modelos base e como realizar fit e predict dos modelos
    - `posmodel/`: Execução de pós processamentos, normalmente as funções de combinação de ensemble
