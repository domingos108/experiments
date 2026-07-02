# CLAUDE.md — Projeto de Dissertação: Séries Temporais (Single Models vs. Sistemas Híbridos Residuais)

Este arquivo é lei operacional para qualquer sessão do Claude Code neste repositório. As regras aqui descritas resultam de decisões explícitas do pesquisador responsável e **não devem ser reinterpretadas ou "corrigidas" por iniciativa própria**. Em caso de dúvida sobre uma regra, pare e pergunte — não assuma.

---

## 1. Build & Environment

Ambiente Windows, shell primário PowerShell. Existe um `.venv` já provisionado na raiz do projeto.

```powershell
# Ativar o ambiente virtual (sempre antes de qualquer execução Python)
.\.venv\Scripts\Activate.ps1

# Instalar/atualizar dependências de terceiros
pip install -r requirements.txt

# Instalar o pacote local `expts` (src/) em modo editável — necessário para
# que `import config`, `from model import ...`, `from input import ...` etc.
# resolvam corretamente a partir de qualquer notebook ou script
pip install -e .
```

Regras de execução:
- **Nunca** execute scripts Python de dentro de `src/` com `cd src` — os imports (`import config`, `from model import generics`) assumem `src/` no `PYTHONPATH`/`sys.path`, não o diretório de trabalho. Use sempre a raiz do projeto como cwd.
- Notebooks (`notebook/**/*.ipynb`) rodam via Jupyter/papermill e dependem do mesmo `PYTHONPATH`. Ao orquestrar execução em lote, replique o padrão de `run_full_baseline.py` (injeção de `SRC` + `ROOT` em `PYTHONPATH` do subprocesso), não invente um mecanismo novo.
- Scripts `run_*_baseline.py` na raiz de `src/` (`run_arima_base.py`, `run_zhang_baseline.py`, `run_khashei_bijari.py`, `run_nolic_baseline.py`) são executados diretamente com `python src/run_<nome>.py` a partir da raiz do projeto, com o venv ativo.

---

## 2. Estrutura de Diretórios Alvo

| Caminho | Papel | Regras |
|---|---|---|
| `data/raw/` | Fonte única de verdade dos dados brutos. Um `.txt`/`.csv` por série, coluna única `y`. | **Somente leitura** por qualquer script/notebook. Nunca gravar aqui. Nova série → arquivo novo + entrada em `BASE_INFORMATION` (`src/config.py`). |
| `data/result/<experiment_id>/` | Artefatos `.pkl` de cada execução de modelo (`<serie>_<horizon><model_name>.pkl`), via `generics.save_result`/`format_names`. | `experiment_id` é a unidade de isolamento entre linhas de experimento. Nomenclatura de `experiment_id` é responsabilidade manual do pesquisador (ver Seção 3.2). |
| `data/result/comparisons/` | CSVs finais de comparação entre baselines de reprodução de literatura (Zhang, Khashei-Bijari, NoLiC), gerados pelos `run_*_baseline.py`. | Saída derivada — nunca editar manualmente, apenas regenerar via script. |
| `data/result/exploratory_analysis/` | Artefatos descartáveis de prova de conceito (atualmente: Informação Mútua vs. CCF). **Proposital e permanentemente fora do versionamento Git.** | Não tratar como pipeline oficial. Não referenciar a partir de código de produção/baseline. |
| `results/` (raiz, fora de `data/`) | Relatório agregado final (`baseline_metrics.csv`, `baseline_metrics_detail.csv`, `plots/`), produzido por `src/utils/export_metrics_to_csv.py`. | Camada de "relatório", não de "artefato de experimento". |

---

## 3. Rigor Científico das Baselines (Regras Imutáveis)

O projeto já possui 5 arquiteturas de baseline validadas e reproduzíveis. **Qualquer refatoração estrutural deve preservar exatamente o comportamento numérico destas 5 esteiras.**

| Arquitetura | Notebook | Wrapper | `.pkl` |
|---|---|---|---|
| ARIMA (linear) | `notebook/single_models/arima_exec.ipynb` | `model.arima.exec_training_testing` (execução direta, sem `GridSearch`) | `<serie>_1arima.pkl` |
| MLP (single) | `notebook/single_models/mlp_exec.ipynb` | `SKlearnModel` via `grid_search_exp` | `<serie>_1mlp.pkl` |
| SVR (single) | `notebook/single_models/svr_exec.ipynb` | `SKlearnModel` via `grid_search_exp` | `<serie>_1svr.pkl` |
| ARIMA-MLP (híbrido aditivo) | `notebook/residual_hydridsystem/arima_mlp.ipynb` | `hybrid_system_exp.Additive` via `grid_search_exp` | `<serie>_1amv1.pkl` |
| ARIMA-SVR (híbrido aditivo) | `notebook/residual_hydridsystem/arima_svr.ipynb` | `hybrid_system_exp.Additive` via `grid_search_exp` | `<serie>_1as.pkl` |

### 3.1 Convivência dos dois esquemas de `test_size`/`lag_size` — NÃO ALTERAR

`config.py` (`TEST_SIZE`/`VAL_SIZE` fracionários + `BASE_INFORMATION`) é a configuração **genérica**, usada pelos notebooks padrão. Os scripts `run_*_baseline.py` usam `SERIES_CONFIG` locais com `test_size` **absoluto**, propositalmente diferente, para reproduzir exatamente os splits dos papers originais (Zhang, Khashei-Bijari, NoLiC).

- `calculate_metrics_v2.ipynb` e a leitura agregada (`metrics.open_fold_result`) **não recalculam splits** — eles só leem métricas já persistidas dentro do `.pkl` (as 10 repetições finais já computadas). A divergência entre os dois esquemas **não contamina** o relatório.
- Os scripts `run_*_baseline.py` são **exceções estritas**, exclusivas de baselines de reprodução de literatura.
- A próxima fase (Feature Selection nos híbridos originais) usa **exclusivamente** `config.py` + a estrutura padrão via notebooks. Nunca introduzir um terceiro esquema de configuração.

### 3.2 `force=False` e idempotência silenciosa — COMPORTAMENTO INTENCIONAL, NÃO ALTERAR

`grid_seach_multiple_bases` pula séries cujo `.pkl` já existe quando `force=False`, sem validar se os `experiment_params` mudaram desde a última execução. **Isso é proposital**: Grid Search de híbridos (especialmente com RFECV na fase futura) pode levar horas, e a capacidade de retomar após uma queda de máquina é um requisito de produtividade, não um bug.

- **Não adicionar hash de configuração ao nome do arquivo.**
- **Não adicionar validação automática de "config mudou, refazer".**
- A responsabilidade de evitar sobrescrita/confusão de parâmetros é do pesquisador, via nomenclatura manual e disciplinada de `experiment_id` nos notebooks (ex.: `chamados_v1`, `chamados_v2_rfecv`). Ao gerar ou sugerir novos notebooks de experimento, sempre proponha um `experiment_id` novo e explícito — nunca reutilize silenciosamente um `experiment_id` existente para uma configuração diferente.

### 3.3 `data/result/exploratory_analysis/` e `src/test_mutual_information.py` — FORA DE ESCOPO DE REFATORAÇÃO

Prova de conceito em andamento (MI vs. CCF sobre resíduos). Não versionado no Git de propósito. **Não tratar como parte do core do projeto** ao refatorar `generics.py`, `input.py` ou os wrappers de modelo. Não mover, não formalizar, não integrar ao pipeline oficial sem instrução explícita.

### 3.4 Metodologia de Comparação Estatística — Regras de Interpretação

- ARIMA e SVR/ARIMA-SVR rodam com `model_exec=1` (determinísticos); MLP/ARIMA-MLP rodam com `model_exec=10` (estocásticos por inicialização de pesos). **Isso é intencional** e reflete a natureza dos algoritmos — não uniformizar `model_exec` entre eles.
- O valor de comparação oficial é a **média das métricas de teste das 10 repetições** (quando aplicável). O critério de sucesso da dissertação é o **ganho percentual de redução de erro (RMSE/MAPE)**, não significância estatística via p-valor. **Não implementar bootstrap sobre resíduos do ARIMA** nem testes de hipótese formais a menos que explicitamente solicitado.
- A classificação de natureza da série (linear/sazonal/caótica/não-linear) é **feita a priori pela literatura de origem de cada dataset** (ex.: `sunspot`/`coloradoRiver` = caóticas/não-lineares; `airlines` = linear/sazonal), não por teste estatístico de linearidade rodado no pipeline. Ao adicionar uma nova série, exigir do pesquisador a classificação de literatura antes de incluí-la em qualquer tabela comparativa por categoria.

### 3.5 `diff_kpss=False` em `mlp_exec.ipynb` — Exceção Intencional

Desde 2026-07-02, `notebook/single_models/mlp_exec.ipynb` usa `experiment_params['diff_kpss'] = False`, por solicitação do orientador do pesquisador (`diff_kpss=True` causava problemas em alguns casos). Este é o comportamento correto atual do baseline MLP — **não é uma divergência a corrigir** em sessões futuras.

---

## 4. Diretriz de Contribuição Futura — Feature Selection Nativo (Sem Monkey Patch)

A próxima fase da dissertação insere seleção de atributos (inicialmente RFECV / Random Forest) **sobre os lags profundos da série de resíduos $e_t$** dos híbridos aditivos (Zhang 2003), isolando os lags não-lineares mais relevantes antes de treinar a rede/SVR. O roadmap prevê evoluir depois para combinações não-lineares de 2–3 estágios (rede recebendo simultaneamente lags da série original, previsão linear e resíduos) — **o espaço de entrada vai crescer**, então o design do seletor deve ser genérico o suficiente para lidar com matrizes de entrada maiores sem reescrita.

**Regra dura: nenhuma implementação de Feature Selection pode usar monkey patch** (reatribuição dinâmica de métodos em `generics.py`, `SKlearnModel`, `Additive` ou `input.py` em tempo de execução). A integração deve ser nativa — alterando/estendendo as classes de forma explícita e rastreável no controle de versão.

### 4.1 Recomendação técnica: `sklearn.Pipeline` é viável — com uma condição não-negociável

Avaliação do código legado (`generics.py`, `hybrid_system_exp.Additive`, `grid_search_exp.GridSearch`):

- `fit_predict_ml_schemma` (usado tanto por `SKlearnModel` quanto por `Additive`) só chama `model.fit(x_train, y_train)` / `model.predict(x)` — **é agnóstico à identidade do `model`**. Substituir o modelo simples por `Pipeline([('selector', RFECV(...)), ('estimator', model)])` funciona sem tocar em `generics.py`.
- `grid_search_exp.is_not_sklearn()` usa `isinstance(model, BaseEstimator)` — um `Pipeline` **é** uma `BaseEstimator`, então a bifurcação existente já trata Pipelines corretamente pelo ramo `clone(model).set_params(**params)`. Isso exige apenas que `model_parameters` nos notebooks passe a usar a convenção `nome_da_etapa__parametro` (ex.: `{'selector__n_features_to_select': [...], 'estimator__hidden_layer_sizes': [...]}`) — convenção nativa do sklearn, não um hack.
- `CustomBaggingRegressor` ([src/model/custom_bagging.py](src/model/custom_bagging.py)) trata seu `estimator` interno de forma genérica via `fit`/`predict` (delegando ao `BaggingRegressor` do sklearn) — compatível com um `Pipeline` como estimador base, sem acesso direto a atributos internos de MLP/SVR.
- A inversão de normalização/diferenciação de **y** (MinMaxScaler + KPSS) ocorre **fora** do `model.fit`/`model.predict`, dentro de `SKlearnModel.fit_predict`/`Additive.fit_predict`/`generics.format_forecats`. Um `Pipeline` padrão do sklearn só transforma **X** — isso está correto e não deve mudar: o `Pipeline` deve encapsular apenas `[seletor_de_features, estimador]`, nunca a transformação de `y`.

**Condição não-negociável para aprovar qualquer Pipeline com seletor CV-based (RFECV):** o parâmetro `cv` do seletor **jamais** pode usar o `KFold` padrão do sklearn (shuffle aleatório). Isso vazaria observações futuras para trás no tempo durante a eliminação recursiva de features — violação direta de Zero Data Leakage (Seção 5.2). Usar `sklearn.model_selection.TimeSeriesSplit` ou um splitter customizado que respeite a ordem cronológica.

**Risco a validar antes de generalizar a solução:** não há evidência no código atual de acesso direto a atributos internos de `MLPRegressor`/`SVR` fora de suas próprias classes — mas isso deve ser reconfirmado pontualmente para cada novo wrapper híbrido (`KhasheiBijariHybrid`, `NoLiCHybrid`) antes de aplicar o mesmo padrão de Pipeline a eles, pois ambos constroem matrizes de entrada combinadas fora do fluxo padrão de `SKlearnModel`.

**Caminho de refatoração recomendado:**
1. Adicionar a `Pipeline` como o próprio objeto `model` passado ao `GridSearch`/`Additive` nos notebooks — nenhuma mudança em `generics.py`.
2. Ajustar apenas os dicionários `model_parameters` dos notebooks para a convenção `etapa__parametro`.
3. Não introduzir um "modo Feature Selection" paralelo em `input.py` — a matriz de lags (`create_windowing`) continua gerando todos os lags candidatos; a seleção acontece inteiramente dentro do `Pipeline`, depois do janelamento.

---

## 5. 🛠️ Orquestração de Skills/Plugins do Claude Code (Obrigatório)

Estas diretrizes têm precedência sobre o comportamento default do agente neste repositório.

### 5.1 Antes de qualquer refatoração estrutural
Antes de alterar a estrutura de `generics.py`, `grid_search_exp.py`, `hybrid_system_exp.py`, `single_ml_model_exp.py` ou `input.py` — ou de introduzir qualquer novo wrapper de experimento — **invocar a skill `superpowers:brainstorming`** e produzir um bloco `<thinking>` explícito de análise investigativa (impacto na esteira de dados, nos 5 baselines da Seção 3, e no plano de Feature Selection da Seção 4) antes de escrever qualquer código.

### 5.2 Antes de finalizar a implementação de um modelo
Antes de considerar qualquer implementação de modelo/wrapper como concluída — em especial qualquer coisa que toque em split treino/val/teste, normalização, ou seleção de features — **invocar a skill `code-review`** internamente, com foco explícito em:
- **Zero Data Leakage**: nenhum `KFold`/`cross_val_score`/`cv` aleatório sobre dados temporais; nenhuma estatística (scaler, seletor, KPSS) ajustada fora do conjunto de treino; nenhuma informação de `df_val`/`df_test` vazando para trás no tempo.
- Conformidade com as regras imutáveis da Seção 3.

Só reportar a tarefa como concluída após esse gate passar.

### 5.3 Limite de complexidade em loops de Grid Search / validação
Se qualquer loop de Grid Search (`GridSearch._search_params`, `GridSearch.execution`, ou equivalente novo) ou função de validação ultrapassar **4 níveis de indentação**, **invocar automaticamente a skill/agente `code-simplifier`** antes de prosseguir, para extrair funções auxiliares e preservar legibilidade — sem alterar o comportamento numérico dos experimentos.
