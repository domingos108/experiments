# Obejetivo do repositório

Executar experimentos para previsão de séries temporais. 

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
