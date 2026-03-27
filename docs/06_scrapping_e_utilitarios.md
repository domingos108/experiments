# 06 — Scraping e Utilitários de Dados

Este documento descreve os scripts de extração e formatação de dados externos, como esses dados se conectam ao pipeline principal de séries temporais, e como atualizar as bases no futuro.

---

## 1. Visão Geral das Fontes de Dados

O projeto utiliza dados de **quatro origens externas distintas**, cada uma com seu próprio script/notebook de ingestão e formatação:

| Fonte | Script | Datasets gerados | Frequência |
|-------|--------|-----------------|------------|
| Prefeitura do Recife — Acidentes de Trânsito | `notebook/recife_acidentes_format.ipynb` | `recifeaccday.txt`, `recifeaccweek.txt`, `recifeaccmonth.txt`, `marecacc.txt` | Diária / Semanal / Mensal |
| SAMU — Chamadas de Emergência | `data/raw/samu/format_samu.ipynb` | `masamu.txt`, `samurec.txt`, `majaboataosamu.txt`, `maolindasamu.txt`, `mapaulistasamu.txt`, `marecifesamu.txt` | Diária |
| ANEEL — Consumo de Energia Elétrica | `data/raw/consumo_energia/format_data.ipynb` | `consumocoformated.txt`, `consumoneformated.txt`, `consumonoformated.txt`, `consumosdformated.txt`, `consumosulformated.txt` | Mensal |
| G1 / Globo — Notícias (experimental) | `notebook/scrapping.ipynb`, `notebook/g1scrapping.py` | Estrutura JSON em memória (sem saída `.txt` padronizada ainda) | Sob demanda |

**Conexão com o pipeline principal:** todos os scripts de formatação produzem arquivos `.txt` com uma única coluna `y` em `data/raw/`, que é exatamente o formato consumido por `input.load_raw_data()`. Os metadados de cada série (frequência, período sazonal, lag) são registrados manualmente em `src/config.py → BASE_INFORMATION` após a criação do arquivo.

---

## 2. Base de Acidentes de Trânsito do Recife (`recife_acidentes_format.ipynb`)

### 2.1 Fonte dos Dados

Arquivos CSV disponibilizados pelo Portal de Dados Abertos da Prefeitura do Recife, um por ano:

```
acidentes-transito-2015.csv
acidentes_2016.csv
acidentes_2017.csv
acidentes_2018.csv
acidentes-2019.csv
```

### 2.2 Fluxo de Transformação

```
CSV por ano (situacao, data, natureza_acidente, ...)
    │
    ▼  Filtrar: situacao == 'FINALIZADA'
    │
    ▼  Agregar: GROUP BY data → COUNT(natureza_acidente)
    │
    ▼  Concatenar todos os anos → df completo
    │
    ▼  Preencher datas faltantes (join com date_range completo)
    │     └─ Imputação: média dos últimos 4 registros do mesmo dia da semana
    │
    ▼  Gerar 3 granularidades:
    ├─ Diária   →  recifeaccday.txt
    ├─ Semanal  →  recifeaccweek.txt  (GROUP BY ano×semana_iso)
    └─ Mensal   →  recifeaccmonth.txt (GROUP BY ano×mês)
    │
    ▼  Suavizar com média móvel 7 dias → marecacc.txt
```

### 2.3 Lógica de Imputação de Datas Faltantes

Para dias sem registros (feriados, falhas de sistema):

```python
# Para cada data faltante, usa a média dos últimos 4 registros
# do mesmo dia da semana (ex: segunda-feira → média das últimas 4 segundas)
last_values = df[(df['date'] < date) & (df['weekday'] == weekday)].iloc[-4:]['y'].mean()
```

### 2.4 Colunas Relevantes do CSV Original

| Coluna | Uso |
|--------|-----|
| `situacao` | Filtro: manter apenas `'FINALIZADA'` |
| `data` | Chave de agrupamento temporal |
| `natureza_acidente` | Contagem (variável alvo `y`) |

---

## 3. Base do SAMU (`data/raw/samu/format_samu.ipynb`)

### 3.1 Fonte dos Dados

Arquivos CSV de chamadas do SAMU (Serviço de Atendimento Móvel de Urgência), coletados a partir de `2022-06-01` para evitar distorções do período COVID-19.

```
/home/.../samudata/*.csv    →   múltiplos arquivos mensais carregados via glob
```

### 3.2 Fluxo de Transformação

```
Múltiplos CSVs mensais (data, hora_minuto, municipio, subtipo, motivo_finalizacao, ...)
    │
    ▼  Normalizar hora_minuto (zero-pad)
    │
    ▼  Padronizar origem_chamado (ex: 'VIA PÚBLICA' → 'VIA PUBL')
    │
    ▼  Concatenar + drop_duplicates
    │
    ▼  Filtros aplicados:
    │     ├─ Excluir: subtipo contém 'COVID'
    │     ├─ Excluir: motivo_finalizacao contém 'DUPLICADA'
    │     └─ Manter: data >= '2022-06-01'
    │
    ▼  GROUP BY data → COUNT(hora_minuto)
    │
    ▼  Garantir continuidade: join com date_range completo (freq='D')
    │
    ▼  Aplicar log + média móvel 7 dias (janela de suavização)
    │
    ├─ masamu.txt   →  série suavizada com log (log + MA7)
    └─ samurec.txt  →  série diária bruta (contagem pura)
```

### 3.3 Sobre os Arquivos `majaboataosamu.txt`, `maolindasamu.txt`, etc.

Esses arquivos correspondem a recortes geográficos por **bairro/região do Recife** (Jaboatão, Olinda, Paulista, Recife), gerados com filtros adicionais na coluna `municipio` — seguindo a mesma lógica do `format_samu.ipynb` mas com `df_all[df_all['municipio'] == 'RECIFE']`, por exemplo.

---

## 4. Base de Consumo de Energia Elétrica (`consumo_energia/format_data.ipynb`)

### 4.1 Fonte dos Dados

Arquivos CSV com dados mensais de consumo de energia por região do Brasil (fonte: ANEEL ou EPE):

```
consumo na região Centro-Oeste.csv
consumo na região Nordeste.csv
consumo na região Norte.csv
consumo na região Sudeste.csv
consumo na região Sul .csv
```

### 4.2 Fluxo de Transformação

Os arquivos já estão no formato mensal com coluna `y`. O notebook aplica principalmente:

1. **Transformação logarítmica** para estabilizar variância:
   ```python
   df['y'] = np.log(df['y'])
   df.to_csv(serie_name.replace('.txt', 'log.txt'), index=False)
   ```
2. **Análise de ACF/PACF** para inspeção visual da autocorrelação e sazonalidade.
3. **Geração de dois arquivos por região:** versão original (`*formated.txt`) e versão log-transformada (`*formatedlog.txt`).

### 4.3 Arquivos Gerados

| Arquivo | Conteúdo |
|---------|----------|
| `consumocoformated.txt` | Consumo mensal — Centro-Oeste (escala original) |
| `consumocoformatedlog.txt` | Idem — log-transformado |
| `consumoneformated.txt` | Consumo mensal — Nordeste |
| `consumonoformated.txt` | Consumo mensal — Norte |
| `consumosdformated.txt` | Consumo mensal — Sudeste |
| `consumosulformated.txt` | Consumo mensal — Sul |

---

## 5. Scraping do Portal G1/Globo (`scrapping.ipynb` e `g1scrapping.py`)

### 5.1 O que faz

Script **experimental** de web scraping da página inicial do G1 para coletar notícias e seus metadados:

```
https://www.globo.com/   →   extrai todos os links <a href>
    │
    ▼  Filtrar: links que contêm '/noticia' e '.ghtml'
    │
    ▼  Para cada link de g1.globo.com:
    │     get_g1globo_page(link)
    │     ├─ titulo     →  div.class='title'
    │     ├─ subtitulo  →  h2.class='content-head__subtitle'
    │     ├─ resumo     →  li.class='mc-summary-card__item'
    │     └─ todos_paragrafos  →  todas as tags <p>
    │
    ▼  Retorna lista de dicionários:
         [{'titulo': ..., 'subtitulo': ..., 'resumo': [...], 'link': ..., 'todos_paragrafos': [...]}, ...]
```

### 5.2 Estado atual e conexão com o projeto

> **Atenção:** Este script está em fase **experimental**. Ele não produz nenhum arquivo `.txt` no formato padrão `y` e não está referenciado em `config.BASE_INFORMATION`. A intenção provável é usar o texto das notícias como **variável exógena** ou contexto qualitativo para análise de séries temporais de demanda de serviços de emergência (SAMU, acidentes).

---

## 6. Utilitário de Análise ACF/PACF (`notebook/acf_analysis.ipynb`)

### 6.1 O que faz

Notebook utilitário para **análise exploratória** de séries temporais antes da modelagem:

- Plota ACF e PACF com intervalo de confiança 95% ($\pm 1.96/\sqrt{N}$) em formato de barras.
- Gera visualização da série temporal em linha.
- Executa teste KPSS para indicar necessidade de diferenciação.
- Salva os gráficos em PDF (`<nome>_acf.pdf`, `<nome>_ts.pdf`).
- Imprime os lags significativos da PACF — útil para definir `lag_size` em `BASE_INFORMATION`.

### 6.2 Uso típico

```python
# Configurar locale pt_BR para formatação de eixos
# Carregar série pelo path
real = pd.read_csv(i['path_data'], names=['actual'])['actual'].values

# Verificar necessidade de diferenciação
from pmdarima.arima import KPSSTest
kps = KPSSTest()
print(kps.should_diff(real[0:-test_size]))

# Gerar gráficos ACF/PACF e plotar série
get_acf(real, test_size, {'name': 'minha_serie', 'label_y': 'Valor', 'label_x': 'Tempo'})
```

---

## 7. Como Atualizar as Bases no Futuro

### 7.1 Atualizar dados de Acidentes de Trânsito do Recife

1. Baixar o novo arquivo anual do portal de dados abertos da Prefeitura do Recife.
2. Adicionar a nova leitura no notebook seguindo o padrão existente:
   ```python
   df_2020 = pd.read_csv('/caminho/para/acidentes_2020.csv')
   daily_2020 = df_2020[df_2020['situacao'] == 'FINALIZADA'].groupby('data')['natureza_acidente'].count().reset_index()
   daily_2020.columns = ['date', 'y']
   ```
3. Incluir `daily_2020` na lista do `pd.concat`.
4. Re-executar todas as células — os arquivos `.txt` em `data/raw/` serão sobrescritos.

### 7.2 Atualizar dados do SAMU

1. Adicionar os novos CSVs mensais na pasta `samudata/`.
2. Atualizar o filtro de data mínima se necessário:
   ```python
   df_sample = df_all[(
       (~df_all['subtipo'].str.contains('COVID', na=False)) &
       (~df_all['motivo_finalizacao'].str.contains('DUPLICADA', na=False)) &
       (df_all['data'] >= '2022-06-01')  # ← ajustar se necessário
   )]
   ```
3. Re-executar o notebook — `masamu.txt` e `samurec.txt` serão atualizados.

### 7.3 Atualizar dados de Consumo de Energia

1. Substituir os CSVs em `data/raw/consumo_energia/` pelos novos arquivos da ANEEL/EPE.
2. Conferir se o nome da coluna `y` já está correto no CSV ou ajustar o `read_csv`.
3. Re-executar o notebook para gerar as versões `*formated.txt` e `*formatedlog.txt`.

### 7.4 Registrar a série atualizada no pipeline

Após gerar o novo `.txt`, **obrigatoriamente** atualizar `src/config.py`:

```python
# 1. Adicionar em BASE_INFORMATION
BASE_INFORMATION['minha_serie.txt'] = {
    "freq": "D",      # frequência temporal
    "m":    7,        # período sazonal
    "lag_size": 7     # ou 'auto'
}

# 2. Incluir na lista de execução
BASE_NAME_LIST = [
    ...
    'minha_serie.txt',
]
```

### 7.5 Validar a série antes de usar

```python
# Usar acf_analysis.ipynb para confirmar:
# 1. Série não tem valores NaN
# 2. Lag sazonal correto (pico na PACF)
# 3. Necessidade de diferenciação (teste KPSS)
# 4. Tamanho mínimo da série (recomendado: > 100 observações)

import pandas as pd
df = pd.read_csv('data/raw/minha_serie.txt')
print(df.shape)           # verificar tamanho
print(df['y'].isna().sum())  # verificar NaNs
print(df['y'].describe())    # estatísticas básicas
```

---

## 8. Diagrama de Fluxo: Dados Externos → Pipeline

```
Fontes Externas
  │
  ├─ Prefeitura Recife (CSV por ano)
  │     └─ recife_acidentes_format.ipynb
  │           ├─ recifeaccday.txt
  │           ├─ recifeaccweek.txt
  │           ├─ recifeaccmonth.txt
  │           └─ marecacc.txt
  │
  ├─ SAMU (CSV mensais)
  │     └─ data/raw/samu/format_samu.ipynb
  │           ├─ masamu.txt
  │           ├─ samurec.txt
  │           └─ ma<bairro>samu.txt (por município)
  │
  ├─ ANEEL/EPE (CSV por região)
  │     └─ consumo_energia/format_data.ipynb
  │           └─ consumo<regiao>formated[log].txt
  │
  └─ G1 Globo (Web Scraping)
        └─ scrapping.ipynb / g1scrapping.py
              └─ [experimental — sem saída padronizada ainda]

Todos os .txt acima
        │
        ▼
data/raw/<serie>.txt  →  coluna única 'y'
        │
        ▼
src/config.py (BASE_INFORMATION + BASE_NAME_LIST)
        │
        ▼
input.load_raw_data()  →  Pipeline completo de modelagem
```
