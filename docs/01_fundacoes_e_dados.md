# 01 — Fundações e Dados

Este documento descreve a camada de configuração e o pipeline de ingestão de dados do projeto de previsão de séries temporais.

---

## 1. Configuração Global (`src/config.py`)

### Constantes de Caminho

| Constante         | Descrição |
|-------------------|-----------|
| `ROOT_PATH`       | Raiz do projeto, encontrada automaticamente subindo a árvore de diretórios a partir de `config.py` até localizar `setup.py`. |
| `RAW_DATA_PATH`   | `<ROOT_PATH>/data/raw/` — diretório onde todos os arquivos de séries temporais brutas estão armazenados. |
| `MODEL_DATA_PATH` | `<ROOT_PATH>/data/result/` — diretório para persistência dos resultados dos modelos. |

### Frações de Split Padrão

| Constante   | Padrão | Descrição |
|-------------|--------|-----------|
| `TEST_SIZE` | `0.1`  | Fração da série reservada para teste. |
| `VAL_SIZE`  | `0.1`  | Fração da série reservada para validação. |

> Estes são valores globais padrão. Cada experimento passa seu próprio dicionário `exec_config` que pode sobrescrevê-los.

### Registro de Datasets (`BASE_INFORMATION`)

Todo dataset utilizado no projeto deve ser registrado em `BASE_INFORMATION` com as seguintes chaves:

```python
BASE_INFORMATION = {
    'minha_serie.txt': {
        "freq":     "MS",   # Alias de frequência do pandas
        "m":        12,     # Período sazonal (usado para diferenciação)
        "lag_size": 12      # Tamanho da janela de entrada, ou 'auto'
    },
    ...
}
```

| Chave      | Tipo               | Descrição |
|------------|--------------------|-----------|
| `freq`     | `str`              | Alias de offset do pandas (ex: `"D"` diário, `"MS"` mensal, `"QE"` trimestral, `"YE"` anual). Utilizado por modelos que necessitam da frequência temporal. |
| `m`        | `int`              | Período sazonal. Utilizado como ordem de diferenciação ao transformar uma série não estacionária. |
| `lag_size` | `int` ou `"auto"` | Número de defasagens (lags) passadas ao modelo. `"auto"` ativa a seleção automática via PACF (até lag 20). |

---

## 2. Formato dos Dados Brutos

Todos os datasets ficam em `data/raw/` como arquivos de texto CSV (`.txt` ou `.csv`).

**Formato obrigatório:** uma única coluna chamada `y`, com uma observação por linha.

```
y
120.5
134.2
119.8
...
```

- Nenhuma coluna de data é necessária para o treinamento; a série é tratada como uma sequência pura.
- Os valores de `freq` e `m` em `BASE_INFORMATION` fornecem a semântica temporal consumida pelos modelos baseados em statsmodels.

---

## 3. Pipeline de Ingestão (`src/input/input.py`)

O ponto de entrada principal é `open_format_train_val_test(base_name, exec_config)`.

### Parâmetros do `exec_config`

| Chave         | Tipo                        | Descrição |
|---------------|-----------------------------|-----------|
| `horizon`     | `int`                       | Horizonte de previsão (número de passos à frente). Afeta o janelamento e a ordem de diferenciação. |
| `normalize`   | `bool`                      | Se `True`, aplica MinMaxScaler ajustado no split de treino. |
| `lag_size`    | `int` \| `"auto"`           | Tamanho da janela de entrada. Sobrescreve o valor do registro quando passado diretamente. |
| `diff_kpss`   | `bool`                      | Se `True`, executa teste KPSS; aplica diferenciação sazonal se a série for não estacionária. |
| `type_filter` | `None` \| `"ma"` \| `"db4"` | Filtro de pré-processamento aplicado antes do janelamento (ver detalhes abaixo). |
| `test_size`   | `float`                     | Fração das observações para o conjunto de teste (ex: `0.1`). |
| `val_size`    | `float`                     | Fração das observações para o conjunto de validação (ex: `0.1`). |

### Etapas de Processamento

```
CSV Bruto
  │
  ▼
load_raw_data()          # pd.read_csv de data/raw/
  │
  ▼
exec_filter()            # Suavização/denoising opcional
  │  ├─ None   → sem processamento
  │  ├─ "ma"   → média móvel com janela = lag_size
  │  └─ "db4"  → denoising por wavelet Daubechies-4, soft-threshold (nível 2)
  │
  ▼
StationarityKPSS         # (somente se diff_kpss=True)
  │  └─ não estacionária → diferenciação sazonal de ordem m
  │
  ▼
MinMaxScaler             # (somente se normalize=True, ajustado apenas no treino)
  │
  ▼
create_windowing()       # Constrói a matriz supervisionada de lags
  │  colunas: [lag_1, ..., lag_N, actual]
  │
  ▼
Divisão Treino / Val / Teste
  ├─ df_train  →  linhas  0                       : -(test_size + val_size)
  ├─ df_val    →  linhas  -(test_size + val_size)  : -test_size
  └─ df_test   →  linhas  -test_size               : fim
```

### Saída: `OpenDataOutput`

A função retorna um objeto `OpenDataOutput` com os seguintes atributos:

| Atributo            | Descrição |
|---------------------|-----------|
| `ts_univariate`     | Array numpy processado (após filtro + diferenciação). |
| `df_train`          | DataFrame de treino janelado. |
| `df_val`            | DataFrame de validação janelado. |
| `df_test`           | DataFrame de teste janelado. |
| `min_max_scaler`    | Instância do scaler ajustado (`None` se `normalize=False`). |
| `test_size`         | Número absoluto de amostras de teste. |
| `val_size`          | Número absoluto de amostras de validação. |
| `is_stationary`     | `bool` — se a série de treino passou no teste KPSS. |
| `original_ts`       | Valores brutos de `y` antes de qualquer transformação. |
| `lag_size_formated` | Lag resolvido como inteiro (mesmo quando `"auto"` foi solicitado). |
| `freq`              | String de frequência do pandas vinda do registro. |
| `m`                 | Período sazonal vindo do registro. |

Os resultados também podem ser acessados via `.sequential_return()` (tupla) ou `.dict_return()` (dicionário).

---

## 4. Seleção Automática de Lag (`get_max_lag_to_consider`)

Quando `lag_size = "auto"`, o lag é determinado por:

1. Cálculo da PACF até lag 20 sobre a porção de treino da série.
2. Cálculo do limiar do intervalo de confiança 95%:

$$\text{limiar} = \frac{1{,}96}{\sqrt{N_{treino}}}$$

3. Seleção do **maior índice de lag** cujo valor de PACF excede o intervalo de confiança.

---

## 5. Adicionando um Novo Dataset

1. Coloque o arquivo em `data/raw/` como `minha_serie.txt`, com uma única coluna `y`.
2. Registre-o em `BASE_INFORMATION` dentro de `src/config.py`:
   ```python
   'minha_serie.txt': {"freq": "MS", "m": 12, "lag_size": "auto"}
   ```
3. Opcionalmente, adicione-o à `BASE_NAME_LIST` para incluí-lo nas execuções em lote.
