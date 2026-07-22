# PLANO_ARQUITETURA.md — Feature Selection Nativo para os Sistemas Híbridos Residuais

Este documento traduz a diretriz da Seção 4 do [CLAUDE.md](CLAUDE.md) em um plano de arquitetura concreto: como o `TimeSeriesFeatureSelector` será injetado via `sklearn.Pipeline` nos híbridos ARIMA-MLP/ARIMA-SVR, qual arsenal de algoritmos será construído, em que ordem, e quais são as próximas tarefas.

---

## 1. A Arquitetura Validada

### 1.1 Onde o Pipeline entra

`Additive.fit_predict()` ([src/model/hybrid_system_exp.py:85](src/model/hybrid_system_exp.py#L85)) delega o treino ao objeto `self.model` via `generics.fit_predict_model()` → `fit_predict_ml_schemma()`, que só chama `model.fit(x_train, y_train)` / `model.predict(x)`. Isso é **agnóstico à identidade de `model`** — trocar o estimador simples por um `Pipeline` não exige nenhuma mudança em `generics.py` nem em `hybrid_system_exp.py`.

```python
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from model.feature_selection import TimeSeriesFeatureSelector

model = Pipeline([
    ('selector', TimeSeriesFeatureSelector(strategy='mutual_info', k=10)),
    ('estimator', MLPRegressor(activation='identity')),
])
```

`grid_search_exp.GridSearch` já trata isso corretamente sem alteração: `is_not_sklearn(model)` usa `isinstance(model, BaseEstimator)`, e `Pipeline` **é** uma `BaseEstimator` — cai no ramo `clone(self.model).set_params(**params)`, que resolve `stepname__param` nativamente via `get_params(deep=True)`.

### 1.2 Como fica o `model_parameters` do Grid Search

Convenção sklearn padrão, sem hack:

```python
model_parameters = {
    'selector__strategy': ['mutual_info'],       # fixo por notebook (não misturar estratégias no mesmo grid)
    'selector__k': [5, 10, 15, 20],               # nº de lags selecionados — varia por método
    'estimator__hidden_layer_sizes': [(10,), (20,), (50,)],
}
```

`ParameterGrid` em `GridSearch._search_params()` já faz o produto cartesiano `selector__* × estimator__*` automaticamente — a busca cruzada entre seletor e rede/SVR **já é suportada pelo código existente**, zero mudança em `grid_search_exp.py` para isso.

### 1.3 Obstáculo real encontrado: `lag_size` é sobrescrito pelo `GridSearch`

`GridSearch._search_params()` e `GridSearch.execution()` ([src/model/grid_search_exp.py:57](src/model/grid_search_exp.py#L57) e [:103](src/model/grid_search_exp.py#L103)) fazem:

```python
experiment_params['lag_size'] = config.BASE_INFORMATION[self.base_name]['lag_size']
```

Isso **sobrescreve incondicionalmente** qualquer `lag_size` definido no notebook — inclusive o que os híbridos herdam via `input_linear_info()`. O `lag_size` em `BASE_INFORMATION` é o mesmo usado pelos 5 baselines (tipicamente 7, 12 ou `'auto'` via PACF), pensado para janelas curtas de modelo único — **não** para expor 20–30 lags profundos do resíduo $e_t$ ao seletor, como planejado no Objetivo 4 do roadmap.

Isso conflita com a regra imutável da Seção 3.1 do CLAUDE.md ("`config.py` é fonte única, nunca introduzir um terceiro esquema"), então a solução não pode ser um config paralelo. **Correção mínima e aditiva, dentro de `config.py`:**

```python
# config.py — chave opcional, só usada quando presente
BASE_INFORMATION = {
    'sunspot.txt': {"freq": "YE", 'm': 1, 'lag_size': 'auto', 'fs_lag_size': 30},
    ...
}
```

```python
# grid_search_exp.py — fallback de 2 linhas, não quebra nenhum experimento existente
base_lag_info = config.BASE_INFORMATION[self.base_name]
experiment_params['lag_size'] = base_lag_info.get('fs_lag_size', base_lag_info['lag_size'])
```

Sem `fs_lag_size` definido, o comportamento é idêntico ao atual (`.get()` cai no `lag_size` de sempre) — os 5 baselines da Seção 3 do CLAUDE.md permanecem numericamente intocados. Esta é a **Tarefa 1** do roadmap (Seção 3 abaixo).

### 1.4 Onde a transformação de `y` continua vivendo

Sem mudança: a inversão de normalização/diferenciação de `y` (MinMaxScaler + KPSS) permanece **fora** do `Pipeline`, dentro de `Additive.fit_predict()`/`generics.format_forecats`. O `Pipeline` encapsula estritamente `[selector, estimator]` sobre `X`.

### 1.5 Tarefa 3.1 — Reversão para `SelectFromModel` nativo + registro de features selecionadas

A Tarefa 3 implementou `rf_embedded`/`lasso` como top-k uniforme (mesmo contrato de `f_test`/`mutual_info`, `k` fixo pesquisável). A pedido do orientador, a Tarefa 3.1 reverte isso: `rf_embedded`/`lasso` passam a decidir o nº de features **pelo próprio modelo** (`SelectFromModel`), não mais por um `k` de grid search.

**Threshold do `SelectFromModel`: `threshold=None` (default do sklearn), não um valor manual.** Investigação do código-fonte do sklearn (`sklearn.feature_selection._from_model._calculate_threshold`) confirma que o default já resolve exatamente para a filosofia que motivou essa reversão — "nº de features emerge do ajuste", sem inventar um corte arbitrário:
- `LassoCV`: detectado via `"Lasso" in estimator.__class__.__name__` → threshold = `1e-5` (mantém qualquer coeficiente não zerado pela esparsidade L1 — a única coisa que faz sentido pedir de um seletor L1).
- `RandomForestRegressor`: sem esse padrão → threshold = `'mean'` das importâncias (uso canônico de `SelectFromModel` com modelos de árvore).

**`TimeSeriesFeatureSelector` continua sendo a única classe pública** (não uma troca por `SelectFromModel` cru no Pipeline): para `rf_embedded`/`lasso`, `fit()` cria e ajusta um `SelectFromModel(estimator, threshold=None)` internamente e converte `get_support()` em `self.selected_indices_` (mesmo atributo/contrato usado por `f_test`/`mutual_info` e por `transform()`). Isso mantém uma extração pós-hoc uniforme entre as 4 estratégias (ver abaixo) e não exige tocar `transform()`. `k` deixa de ser lido nessas duas estratégias; os notebooks correspondentes removem `selector__k` de `model_parameters`.

**Registro do nº/índices de features selecionadas — decisão de design (Tarefa 3.1, validada empiricamente antes de implementar):**

`Additive`/`SKlearnModel` guardam `self.model` (o `Pipeline`), e `generics.fit_predict_ml_schemma` faz `model.fit(...)` **mutando esse objeto in place**. `GridSearch.execution()` serializa a própria instância `Additive`/`SKlearnModel` inteira (`predict_results.append({'experiment': model_exp_test, ...})` → `generics.save_result`) — ou seja, **o seletor já ajustado (com `selected_indices_`) já sobrevive dentro do `.pkl` de cada execução, hoje, sem qualquer mudança em `generics.py`**. Confirmado por round-trip real de pickle (`save_result`/`open_saved_result`) antes de decidir a abordagem.

Por isso a Parte C da Tarefa 3.1 é um **script de extração pós-hoc**, não uma mudança na gravação: `src/utils/export_selected_features.py`, mesmo esqueleto de `src/utils/export_metrics_to_csv.py` (`parse_pkl_path`, varredura de `.pkl`, CLI `--result-dir`/`--output`). Para cada repetição, lê `entry['experiment'].model.named_steps['selector']` (quando existir — `.pkl` sem seletor, como os 5 baselines, são ignorados com aviso, não erro) e produz:
- `results/<experiment_id>/selected_features.csv` — agregado (média/desvio de `n_features_selected` por série × modelo), mesmo padrão de `aggregate_mean` em `export_metrics_to_csv.py`.
- `results/<experiment_id>/selected_features_detail.csv` — uma linha por repetição: `experiment_id, serie, modelo, repeticao, strategy, n_features_selected, n_features_total, selected_indices, selected_lag_names`.

`selected_lag_names` mapeia índice → nome de coluna (`lag_k`) sem precisar reconstruir o windowing: como `create_windowing`/`input.open_format_train_val_test` sempre nomeiam as colunas `lag_L, lag_{L-1}, ..., lag_1` (L = nº total de features, horizon=1 em todos os notebooks FS), a coluna de índice posicional `j` corresponde a `lag_{L-j}`.

**Variabilidade entre repetições (`model_exec=10`) é esperada e não suprimida.** `TimeSeriesFeatureSelector(strategy='rf_embedded')` já usa `random_state=None` por padrão desde a Tarefa 3 (nenhum notebook fixa `selector__random_state`) — cada uma das 10 repetições re-ajusta o `RandomForestRegressor` interno do zero (`clone(self.model)` por repetição em `GridSearch.execution()`), então o nº de features selecionadas pode variar naturalmente entre repetições. Isso é o dado que o orientador quer observar, por isso fica visível linha-a-linha no `_detail.csv`, sem agregação que esconda a variância. `lasso` (`LassoCV` com `TimeSeriesSplit` determinístico sobre os mesmos dados de treino) tende a ser estável entre repetições — a comparação entre a variância de `rf_embedded` e a estabilidade de `lasso` já é, por si, um resultado a discutir na dissertação.

### 1.6 Tarefa 3.4 — Histórico completo do Grid Search (combinação × erro de validação)

`GridSearch._search_params()` já calcula, para cada combinação do grid, a média (`np.mean`) das `model_exec` repetições internas de erro de validação — mas guarda tudo em variáveis locais (`target_list_mean_metrics`, `list_params`) e retorna só o argmin (a combinação vencedora). Isso impede provar retroativamente que combinações não-vencedoras (ex. `selector__k=15/20` num grid ampliado) foram genuinamente avaliadas, e impede gráficos padrão de FS (erro de validação vs. `k`/nº de features) sem re-rodar do zero (achado da Tarefa 3.3).

**Decisão de design (brainstorming, Seção 5.1 do CLAUDE.md):**

- **Mesmo `.pkl`, nova chave — não um artefato paralelo.** `predict_results` continua sendo uma `list` de `dict` (nenhum leitor existente — `metrics.py`, `export_metrics_to_csv.py`, `export_selected_features.py` — muda de comportamento: todos acessam chaves conhecidas via `.get()`/indexação direta, nenhum valida o conjunto de chaves, então uma chave nova é ignorada silenciosamente pelos leitores antigos). Um artefato paralelo teria risco zero mas exigiria uma convenção de nomenclatura nova sem necessidade — o histórico já nasce junto do resultado que ele explica.
- **Anexado em TODAS as repetições de `predict_results`** (não só a de índice 0). O mesmo histórico fica duplicado `model_exec` vezes (ex. 10x) dentro do `.pkl`, mas o volume de hoje é pequeno (grid típico de FS: ~15 combinações × floats) — a duplicação custa poucos KB. Preferido a anexar só na entrada 0 porque evita uma leitura assimétrica ("cheque só a primeira entrada") — todo consumidor que iterar `predict_results` vê a mesma chave em toda entrada, sem caso especial. Reavaliar se `rfecv`/`perm_importance` (Tarefa 4) inflarem o histórico.
- **Formato de cada entrada:** `{'params': dict, 'val_metric_mean': float, 'val_metric_std': float, 'val_metric_reps': list[float], 'val_metrics_reps': list[dict]}` — `params` e a média já eram calculados; `val_metric_std`/`val_metric_reps` (as `model_exec` repetições internas cruas) já existiam em memória e eram descartados. `val_metrics_reps` (achado de code-review — ângulo altitude) captura o dict **completo** de `val_metrics` por repetição (MSE/RMSE/MAE/MAPE/theil/ARV/IA/POCID, tudo que `metrics.gerenerate_metric_results` já calcula), não só `RMSE` — mesmo argumento "já está em memória, capturar é grátis", só que aplicado à dimensão da métrica também, evitando a mesma parede de "não recuperável" se um dia for preciso um gráfico de MAE/MAPE vs. hiperparâmetro.
- **`save_grid_history=True` por padrão** (novo parâmetro aditivo no fim da assinatura de `GridSearch.__init__`, não desloca nenhum posicional existente). Mudança estritamente aditiva (nenhum valor existente alterado, só uma chave nova), volume pequeno hoje, consistente com o padrão já usado no projeto (`n_features_in_` da Tarefa 3.1 também nasceu sempre-ligado). `save_grid_history=False` deixa o `.pkl` byte-a-byte idêntico ao comportamento anterior (chave nunca é adicionada).
- **Zero Data Leakage por construção:** a captura reusa exatamente `metrics_results.get(self.group_metrics_name, ...)` — a mesma expressão já usada pelo código existente, e `self.group_metrics_name` é fixado como `'val_metrics'` no `__init__` (nunca `'test_metrics'`). Não há caminho de código novo que possa ler teste.
- **`load_grid_search_history` mora em `src/model/grid_search_exp.py`, não em `src/utils/`** (achado de code-review — ângulo altitude, corrigido após a implementação inicial): o leitor decodifica um formato que só `GridSearch` produz, então fica colocado com quem o produz — mesmo precedente de `metrics.open_fold_result` viver em `src/metrics/metrics.py` ao lado de `gerenerate_metric_results`. `src/utils/` continua reservado para os scripts CLI que varrem **muitos** `.pkl` (`export_metrics_to_csv.py`, `export_selected_features.py`, `compare_fs_vs_baseline.py`), um propósito diferente.
- **Bug real corrigido antes de considerar a tarefa concluída (achado de code-review — ângulo line-by-line):** para `model_class_exp` **não-sklearn** (LSTM/NBEATS/ELM/SCN/etc., via `neural_forecast_exp.py`/`perturbative_neural_forecast.py`), `experiment_params['model_actual_config']` é o **mesmo objeto** `dict` que veio de `ParameterGrid` — e esses wrappers mutam esse dict em `fit_predict()` (injetam `random_seed`, `input_size`, `logger`, etc.). Um snapshot de `params` feito *depois* do loop interno capturaria essa poluição em vez dos hiperparâmetros reais do grid. Corrigido tirando o snapshot (`dict(params)`) **antes** do loop interno rodar, uma vez por combinação — não afeta os 5 baselines nem os notebooks de FS (todos usam `Pipeline` sklearn, que não sofre essa mutação), mas é essencial para qualquer wrapper não-sklearn treinado através da mesma `GridSearch` compartilhada.

**Limitação aceita:** o histórico de `chamados_v4_fs_ftest` (já rodado antes desta instrumentação) não é recuperável — só existiu como variável local durante uma execução já finalizada. A instrumentação vale para rodadas futuras; recapturar o histórico dessa rodada específica exige re-rodar o grid.

### 1.7 Tarefa 4 — `rfecv` (último método obrigatório do arsenal)

**Decisões de design (brainstorming, confirmadas com o pesquisador antes de implementar):**

- **`RFECV(RandomForestRegressor(random_state=...), cv=TimeSeriesSplit(n_splits=3), step=1, min_features_to_select=1)`** — mesma convenção de `cv` cronológico obrigatório já usada pelo Lasso (Seção 1.5), reforçada aqui por ser o método mais caro e mais fácil de configurar `cv` errado do arsenal.
- **`support_`/`np.where(...)` mapeiam para `selected_indices_`** exatamente como `SelectFromModel.get_support()` já faz para `rf_embedded`/`lasso` — confirmado com dados reais antes de assumir compatibilidade automática com `export_selected_features.py`.
- **`min_features_to_select=1` é um piso garantido pelo próprio `RFECV`** (nunca retorna 0 features) — diferente do Lasso, **não precisa** do mecanismo de fallback de zero-features da Tarefa 3.1.
- **Achado real do pré-check (não hipotético): `RFECV` levanta `ValueError` com `n_features_in_ < 2`** — RFE não faz sentido conceitualmente com 1 única feature candidata (não há o que eliminar recursivamente). Caso real: `austres.txt` (`N_Features_Total=1`, Seção 1.3/Tarefa 3.2). Decisão confirmada com o pesquisador: guarda defensiva dentro de `_select_via_rfecv` (mantém a única feature trivialmente, sem chamar `RFECV`, com `fallback_triggered_=True`) — mesmo invariante das outras 4 estratégias (nunca crasha, pelo menos 1 feature sempre sobrevive), em vez de excluir `austres.txt` só deste notebook (que deixaria a classe insegura para qualquer uso futuro fora dele).
- **`k` NÃO é lido por `rfecv`** — mesma convenção de `rf_embedded`/`lasso` (Tarefa 3.1/3.9): o número de features emerge da CV interna, não de um hiperparâmetro de grid externo. `arima_mlp_rfecv.ipynb` nasce **sem** `selector__k` no `model_parameters` (não precisa da limpeza que `rf_embedded`/`lasso` precisaram na Tarefa 3.9).
- **Custo computacional medido (não estimado por extrapolação teórica)** antes de finalizar o notebook: 1 fit isolado em dados sintéticos do tamanho real de cada série — `airlines` (80×20) 5.5s, `coloradoRiver` (568×16) 11.7s, `sunspot` (216×9) 2.9s. Extrapolado para produção (3 combinações de `hidden_layer_sizes` × `model_exec=10` na busca + 10 no refit final = 40 fits/série): **≈13-14 minutos para as 4 séries** — viável para "Run All" manual, sem necessidade de `step`/`min_features_to_select` mais agressivo.
- **Escopo estritamente `ARIMA-MLP` (`Additive`)** — `ARIMA-SVR`, `MLP`/`SVR` single ficam para depois de `rfecv` validado aqui, conforme instrução explícita da Tarefa 4.

### 1.8 Tarefa 5 — Generalização para `MLP` single (`SKlearnModel`), primeira família fora de `Additive`

**Pré-checks confirmados com evidência real (não suposição), antes de criar qualquer notebook:**

- **`SKlearnModel` é agnóstico à identidade de `model` na mesma forma que `Additive`** — `fit_predict_ml_schemma` (`src/model/single_ml_model_exp.py`) só chama `model.fit(x_train, y_train)`/`model.predict(x)`, exatamente como `generics.fit_predict_ml_schemma` usado por `Additive`. Confirmado com um teste de integração real (`tests/model/test_single_ml_model_exp.py::TestSKlearnModelAcceptsPipeline`) rodando `Pipeline([selector, MLPRegressor])` fim-a-fim através de `GridSearch`/`SKlearnModel`, sem nenhuma mudança em `single_ml_model_exp.py`/`grid_search_exp.py`.
- **`lag_size='auto'` resolve para os MESMOS valores no contexto MLP single que no híbrido ARIMA-MLP** — medido diretamente (não assumido): `airlines`=20, `austres`=1, `coloradoRiver`=16, `sunspot`=9, idêntico nas 4 séries. Isso não era garantido a priori (`get_max_lag_to_consider` usa PACF sobre a série que de fato chega em `open_format_train_val_test`, que difere entre os dois contextos — série bruta no single-model vs. resíduo do ARIMA no híbrido) — coincidência empírica, não estrutural, e testada (`test_lag_size_auto_resolves_to_same_value_as_arima_mlp_hybrid`) para detectar se algum dia divergir.
- **`df_train` tem mais linhas no single-model que no híbrido** para a mesma série (ex. `airlines`: 96 vs. 80) — o híbrido perde linhas extras por causa do janelamento duplo implícito (o resíduo do ARIMA já é truncado por `lag_size_formated` dentro de `input_linear_info`, depois passa por outro janelamento dentro de `fit_predict_model`). Diferença estrutural esperada, não um bug.
- **`SKlearnModel` não depende de nenhum modelo linear pré-treinado** (ao contrário de `Additive`, que precisa do `.pkl` do ARIMA via `linear_model_name`) — os 5 notebooks de `MLP` single não têm a célula de "copiar o ARIMA pré-treinado" que os notebooks de `ARIMA-MLP` têm (uma célula a menos: 6 em vez de 7).

**Convenção de nomenclatura definida nesta tarefa (referência para `SVR` single/`ARIMA-SVR` nas próximas):**

- **Notebooks**: `notebook/single_models/mlp_<estrategia_fs>.ipynb` (ao lado de `mlp_exec.ipynb`, o baseline protegido) — `mlp_ftest.ipynb`, `mlp_mutual_info.ipynb`, `mlp_rf_embedded.ipynb`, `mlp_lasso.ipynb`, `mlp_rfecv.ipynb`.
- **`model_name`** (sufixo do `.pkl`, sem underscore): `mlpftest`, `mlpmutualinfo`, `mlprfembedded`, `mlplasso`, `mlprfecv` — `GridSearch.__init__` prefixa `f'{horizon}{model_name}'` automaticamente, dando `1mlpftest.pkl` etc. (mesma lógica de `amv1rfembedded` → `1amv1rfembedded.pkl` no híbrido). Confirmado sem colisão com nenhum `.pkl` existente em `data/result/`.
- **`experiment_id`**: `chamados_v4_fs_mlp_<estrategia>` (`chamados_v4_fs_mlp_ftest`, `_mutualinfo`, `_rfembedded`, `_lasso`, `_rfecv`) — o segmento `mlp` distingue explicitamente esta família da família híbrida (`chamados_v4_fs_<estrategia>`, sem `mlp`), já que ambas terão os mesmos 5 nomes de método.
- **Nenhum dos 5 notebooks foi executado** — preparados e validados (sintaxe Python real de cada célula + execução real das células de config/sanity-check, sem tocar `data/result/`), aguardando execução manual.

### 1.9 Portão de validação MLP single + generalização de `compare_fs_vs_baseline.py`

Após a execução manual dos 5 notebooks de FS em `MLP` single (Tarefa 5) e do baseline `1mlp` corrigido (Tarefa 5.1), validação estrutural confirmou: os 20 `.pkl` (5 métodos × 4 séries) têm `activation='logistic'`/`diff_kpss=False` corretos; nenhum arquivo da família híbrida (`amv1*`) contaminou os diretórios `chamados_v4_fs_mlp_*`; a guarda de 1-feature do `rfecv` (Tarefa 4) funcionou corretamente em produção pela primeira vez (`austres`: `fallback_triggered_=True`, sem crash); os 5 baselines protegidos permaneceram intactos (confirmado por `mtime`).

**`src/utils/compare_fs_vs_baseline.py` generalizado** (`build_comparison()` ganhou `baseline_model_name`/`linear_model_name_to_exclude`, com os valores antigos `'1amv1'`/`'1arima'` como default — 100% retrocompatível, testado). Motivo: `BASELINE_MODEL_NAME`/`LINEAR_MODEL_NAME` eram constantes hardcoded específicas da família híbrida; `MLP` single usa baseline `'1mlp'` e não copia nenhum modelo linear para dentro do `result_dir` de FS (`SKlearnModel` não depende de ARIMA pré-treinado), então precisava de `linear_model_name_to_exclude=None` para desligar aquele filtro. `notebook/compare_fs_results_mlp.ipynb` (novo, 3 células, mesmo padrão de `compare_fs_results.ipynb`) é a versão desta família — a lógica de comparação em si não foi duplicada, só a célula de configuração (mesma filosofia "1 notebook por combinação" já usada em todo o projeto, Seção 5). O mesmo padrão (copiar o notebook de comparação, trocar `baseline_model_name`/`fs_variants`) se aplica a `SVR` single/`ARIMA-SVR` quando chegar a vez.

### 1.10 Tarefa 6 — Generalização para `SVR` single (`SKlearnModel`), segunda família fora de `Additive`

Mesma integração `Pipeline([selector, estimador])` da Tarefa 5, agora com `SVR` no lugar de `MLPRegressor` — confirmada com um teste de integração real (`tests/model/test_single_ml_model_exp.py::TestSKlearnModelAcceptsPipelineWithSVR`), sem nenhuma mudança de código. `SVR.get_params()` não colide com `strategy`/`k`/`random_state` do seletor, e o namespace `estimator__*`/`selector__*` do `Pipeline` evitaria colisão de qualquer forma.

**Duas diferenças reais em relação à Tarefa 5 (não um copy-paste ingênuo do template MLP):**

- **`model_exec=1` (determinístico), não 10.** `SVR` é determinístico (CLAUDE.md Seção 3.4, mesma convenção do baseline `svr_exec.ipynb`) — uniformizar com o `model_exec=10` do MLP violaria essa regra explícita.
- **`diff_kpss=True`, não `False`** — `svr_exec.ipynb` usa `diff_kpss=True` (já confirmado sem divergência de artefato na Tarefa 3.9, pré-check 3: `.pkl` persistido bate com o código-fonte atual). `lag_size='auto'` resolve para os MESMOS valores já medidos (`airlines`=20, `austres`=1, `coloradoRiver`=16, `sunspot`=9) — medido com `diff_kpss=True` de verdade, não assumido a partir do valor de `diff_kpss=False` do MLP, porque `get_max_lag_to_consider` roda sobre `ts_univariate` **antes** da diferenciação KPSS (`input.py`), então em teoria `diff_kpss` não deveria afetar esse valor — confirmado empiricamente, não só por leitura de código. `df_train` tem 1 linha a menos que o MLP single para a mesma série (a diferenciação KPSS consome uma linha) — diferença estrutural esperada.

**Grid de hiperparâmetros** extraído de `svr_exec.ipynb`: `C=[10,100,1000]`, `gamma=['auto']`, `kernel=['rbf']`, `epsilon=[0.1,0.01,0.001]`, `tol=[0.001]` — replicado com prefixo `estimator__` nos 5 notebooks novos, mantendo paridade com o baseline.

**Nomenclatura**: `notebook/single_models/svr_<estrategia_fs>.ipynb`, `model_name` = `svrftest`/`svrmutualinfo`/`svrrfembedded`/`svrlasso`/`svrrfecv`, `experiment_id` = `chamados_v4_fs_svr_<estrategia>` — mesmo padrão da Tarefa 5, sem colisão com nenhum `.pkl`/diretório existente. **Nenhum dos 5 notebooks foi executado.**

### 1.11 ⚠️ NOTA PROVISÓRIA (não definitiva) — `gamma='auto'` do `SVR` confunde efeito de FS com largura de kernel

**Esta nota NÃO segue o padrão das Seções 3.5/3.6 do CLAUDE.md (`diff_kpss`/`activation`) — aquelas são decisões fechadas; esta é uma decisão em aberto, sujeita a mudar.**

Achado (família `SVR` single, Tarefa 6): `gamma='auto'` do `sklearn.svm.SVR` é calculado internamente como `1 / n_features`. Como a Feature Selection muda `n_features` por construção, `gamma` muda automaticamente junto com a seleção — o que significa que qualquer diferença de RMSE entre o baseline (todas as features) e uma variante de FS (menos features) mistura dois efeitos que deveriam ser isolados: (a) o efeito genuíno de remover features irrelevantes/redundantes, e (b) o efeito colateral de uma largura de kernel RBF diferente. Isso vale tanto para `SVR` single quanto para este híbrido `ARIMA-SVR` (mesmo estimador, mesmo mecanismo — não é uma surpresa nova aqui, só a mesma limitação se repetindo).

**Decisão do pesquisador: manter `gamma='auto'` por ora, não fixar um valor numérico.** A ser revisada com o orientador. **Se essa decisão mudar no futuro, as famílias `SVR` single (Tarefa 6) e `ARIMA-SVR` (Tarefa 7) precisarão ser re-rodadas** — os `.pkl` já gerados/a gerar não seriam mais comparáveis a uma rodada com `gamma` fixo. Ver também CHECKPOINTS.md, "Pendências conhecidas", para o registro de acompanhamento.

---

## 2. O Arsenal de Algoritmos

`TimeSeriesFeatureSelector` é uma classe única (`BaseEstimator`, `TransformerMixin`) com um parâmetro `strategy` que despacha para implementações internas — permite que o Grid Search trate a *escolha do método* como mais um hiperparâmetro pesquisável, mas na prática cada notebook fixa uma `strategy` por vez para manter a leitura de resultados por família interpretável.

| # | Estratégia | Família / Paradigma | Complexidade & Custo no Grid Search | Valor para a Dissertação | Zero Data Leakage |
|---|---|---|---|---|---|
| 1 | `f_test` (`SelectKBest(f_regression)`) | **Filtro linear.** Ranking por F-estatístico — assume relação linear lag→alvo. | Trivial (sklearn nativo, `fit()` único, sem busca interna). Custo ≈ 0 no Grid Search. | Baseline ingênuo, contraponto direto à CCF já usada no NoLiC — estabelece o "piso" que os métodos não-lineares precisam superar. | Estrutural: `fit()` só vê `X_train` dentro do `Pipeline`; não há CV interna a configurar. |
| 2 | `mutual_info` (`mutual_info_regression`) | **Filtro por Teoria da Informação.** Captura dependência monotônica e não-monotônica via estimação de entropia (KNN), sem assumir linearidade. | Baixa — sklearn nativo, **já importado e validado em `src/test_mutual_information.py`**. Custo moderado (KNN por feature), ainda barato para nossas amostras. | **Alto.** É a promoção direta da prova de conceito exploratória já no repositório (`data/result/exploratory_analysis/mi_vs_ccf_*.csv`) para a arquitetura formal — dá continuidade documentada entre a evidência exploratória e a contribuição da dissertação. | Estrutural, igual ao #1. |
| 3 | `lasso` (`SelectFromModel(LassoCV(...))`) | **Embedded, regularização L1.** Zera coeficientes de lags irrelevantes; robusto a colinearidade entre lags adjacentes de $e_t$ (problema real em séries autocorrelacionadas). | Baixa-média — `LassoCV` aceita `cv=TimeSeriesSplit(...)` nativamente, sem código de busca manual. | Contraponto **linear com seleção esparsa automática** (nº de features não é hiperparâmetro manual, emerge do próprio ajuste) — útil para discutir se a estrutura residual é predominantemente linear-esparsa ou genuinamente não-linear. | **Requer atenção**: `cv` deve ser `TimeSeriesSplit(n_splits=k)` explícito — nunca o `cv=5` default (KFold aleatório). |
| 4 | `rf_embedded` (`SelectFromModel(RandomForestRegressor)`) | **Embedded, redução de impureza.** Importância por Gini/MDI — captura não-linearidade e interação entre lags sem assumir forma funcional. | Média — fit único da RF, sem CV interna obrigatória (a menos que se quantifique estabilidade via múltiplos seeds). | Já exigido como obrigatório — ponto de ancoragem tree-based para comparar com `rfecv` (mesma família, estratégias de seleção diferentes: threshold de importância vs. eliminação recursiva). | Estrutural (fit único em `X_train`); se usar múltiplos seeds para estabilidade, cada um ainda deve treinar só sobre `X_train`. |
| 5 | `rfecv` (`RFECV(estimator=RandomForestRegressor(...), cv=TimeSeriesSplit(...))`) | **Wrapper, minimal-optimal.** Elimina recursivamente a feature menos importante, validando via CV a cada passo — busca o subconjunto mínimo que preserva performance. | **Alta.** Refit a cada eliminação × folds de CV × grid search externo × `model_exec` repetições — é o método mais caro do arsenal. | Já exigido como obrigatório — é o método de referência da literatura de FS wrapper-based para comparar contra os embedded/filtros mais baratos. | **Requer atenção**: `cv=TimeSeriesSplit(n_splits=k)` obrigatório — mesmo risco do Lasso. |
| 6 *(opcional/stretch)* | `perm_importance` (Boruta-lite via `sklearn.inspection.permutation_importance`) | **Wrapper, all-relevant.** Compara importância real contra features "sombra" embaralhadas — filosofia oposta ao RFECV (mantém todas as features estatisticamente relevantes, não só o subconjunto mínimo). | **Muito alta.** Não há Boruta oficial mantido para sklearn moderno; implementação própria multiplica o custo já alto do RFECV. | Interessante para séries caóticas onde múltiplos lags redundantes carregam sinal fraco-mas-real — mas é a única entrada do arsenal sem compromisso de entrega nesta fase. | Permutação deve ocorrer sobre um fold de validação cronológico, nunca embaralhado. |

**Explicitamente fora de escopo agora** (evitar scope creep): busca por algoritmo genético/metaheurístico (ex. `sklearn-genetic-opt`) — mencionado apenas como trabalho futuro, não entra no arsenal desta fase.

### Ordem de implementação (do quick win ao mais caro)

1. **`f_test`** → valida o *plumbing* do `Pipeline` + `GridSearch` com o método mais simples e barato possível, isolando risco de infraestrutura de risco de custo computacional.
2. **`mutual_info`** → mesmo scaffold do #1, só troca a função de score; reaproveita lógica já testada em `test_mutual_information.py`, risco baixo, ganho narrativo alto.
3. **`rf_embedded`** → primeiro método realmente não-linear/embedded; ainda um único fit, sem CV interna a configurar.
4. **`lasso`** → primeiro método com CV interna (`TimeSeriesSplit`) — bom exercício controlado do gate de Zero Data Leakage antes de ir para o método mais caro.
5. **`rfecv`** → implementado por último entre os obrigatórios porque é o mais caro; nesse ponto já existem baselines mais baratos para comparar custo/benefício.
6. *(se houver tempo)* `perm_importance` como adição final, opcional.

---

## 3. Passo a Passo da Implementação (próximas interações)

1. **`/tdd` — Scaffold do `TimeSeriesFeatureSelector` + correção do `lag_size`.**
   Implementar `src/model/feature_selection.py` com `TimeSeriesFeatureSelector(BaseEstimator, TransformerMixin)` suportando `strategy='f_test'` e `strategy='mutual_info'` (métodos #1–#2), seguindo TDD (`superpowers:test-driven-development`). Testes devem cobrir explicitamente: (a) `fit()` nunca recebe índices fora do fold de treino; (b) `set_params`/`clone` funcionam via `GridSearch`; (c) o fallback `fs_lag_size` em `config.py` + `grid_search_exp.py` (Seção 1.3) não altera o `lag_size` de nenhum dos 5 baselines existentes.

2. **Integração ponta-a-ponta em `arima_mlp.ipynb`.**
   Trocar o `MLPRegressor` isolado por `Pipeline([('selector', TimeSeriesFeatureSelector(...)), ('estimator', MLPRegressor(...))])`, ajustar `model_parameters` para a convenção `selector__*`/`estimator__*`, rodar em 2 séries de naturezas opostas (`sunspot.txt` caótica, `airlines.txt` linear/sazonal) com `experiment_id` novo e explícito (ex. `chamados_v2_fs`). Confirmar que o `.pkl` resultante é lido sem erro por `calculate_metrics_v2.ipynb`/`metrics.open_fold_result`.

3. **Adicionar `rf_embedded` e `lasso` ao seletor; rodar grade completa.**
   Estender `TimeSeriesFeatureSelector` com as estratégias #3 e #4, executar `grid_seach_multiple_bases` para ARIMA-MLP e ARIMA-SVR sobre todo `config.BASE_NAME_LIST`, e gerar um CSV comparativo (RMSE/MAPE por série × estratégia) seguindo o padrão de `src/utils/export_metrics_to_csv.py`.

4. **Adicionar `rfecv`; consolidar comparação final e gate de revisão.**
   Implementar a estratégia #5 (`RFECV` com `TimeSeriesSplit`), rodar a comparação final entre os 4 métodos obrigatórios, e — antes de considerar a fase encerrada — invocar `code-review` com foco explícito em Zero Data Leakage nos métodos com CV interna (Lasso e RFECV), conforme a Seção 5.2 do [CLAUDE.md](CLAUDE.md).

---

## 5. Convenção de Nomenclatura de Notebooks por Combinação FS × Arquitetura

`notebook/residual_hydridsystem/arima_mlp.ipynb`/`arima_svr.ipynb` e `notebook/single_models/mlp_exec.ipynb`/`svr_exec.ipynb` permanecem intocáveis — são os baselines da Seção 3 do [CLAUDE.md](CLAUDE.md). Cada combinação de (arquitetura × estratégia de Feature Selection) ganha seu **próprio notebook dedicado**, nunca sobrescrevendo o baseline:

```
<hibrido>_<estrategia_fs>.ipynb          (notebook/residual_hydridsystem/, ex. arima_mlp_ftest.ipynb)
<single_model>_<estrategia_fs>.ipynb     (notebook/single_models/, ex. mlp_ftest.ipynb -- Tarefa 5)
```

`experiment_id`: híbridos usam `chamados_v4_fs_<estrategia>`; famílias single-model usam `chamados_v4_fs_<modelo>_<estrategia>` (ex. `chamados_v4_fs_mlp_ftest`) — o segmento extra (`mlp`, e futuramente `svr`) evita colisão de nomes entre as duas famílias, que compartilham os mesmos 5 nomes de estratégia (Tarefa 5, Seção 1.8).

Exemplos já existentes ou esperados pelo roadmap (Seção 2):

| Notebook | Arquitetura | Estratégia FS | Status |
|---|---|---|---|
| `arima_mlp_ftest.ipynb` | ARIMA-MLP (`Additive`) | `f_test` | Executado (Tarefa 3.2) |
| `arima_mlp_mutual_info.ipynb` | ARIMA-MLP (`Additive`) | `mutual_info` | Executado (Tarefa 3.2) |
| `arima_mlp_rf_embedded.ipynb` | ARIMA-MLP (`Additive`) | `rf_embedded` | Executado e regenerado (Tarefa 3.9) |
| `arima_mlp_lasso.ipynb` | ARIMA-MLP (`Additive`) | `lasso` | Executado e regenerado (Tarefa 3.9) |
| `arima_mlp_rfecv.ipynb` | ARIMA-MLP (`Additive`) | `rfecv` | Implementado, notebook pronto, **não executado** (Tarefa 4) |
| `mlp_ftest.ipynb` | MLP single (`SKlearnModel`) | `f_test` | Notebook pronto, **não executado** (Tarefa 5) |
| `mlp_mutual_info.ipynb` | MLP single (`SKlearnModel`) | `mutual_info` | Notebook pronto, **não executado** (Tarefa 5) |
| `mlp_rf_embedded.ipynb` | MLP single (`SKlearnModel`) | `rf_embedded` | Notebook pronto, **não executado** (Tarefa 5) |
| `mlp_lasso.ipynb` | MLP single (`SKlearnModel`) | `lasso` | Notebook pronto, **não executado** (Tarefa 5) |
| `mlp_rfecv.ipynb` | MLP single (`SKlearnModel`) | `rfecv` | Notebook pronto, **não executado** (Tarefa 5) |
| `arima_svr_ftest.ipynb` | ARIMA-SVR (`Additive`) | `f_test` | Não implementado — fora do escopo da Tarefa 5 |
| `svr_ftest.ipynb` | SVR single (`SKlearnModel`) | `f_test` | Não implementado — fora do escopo da Tarefa 5 |

Quando o roadmap evoluir para as combinações não-lineares de 2–3 estágios (`NonLinear`, Seção 4 do CLAUDE.md), o mesmo padrão se aplica trocando o prefixo do híbrido (ex. `nonlinear_mlp_ftest.ipynb`). Cada `experiment_id` usado nesses notebooks segue a regra da Seção 3.2 do CLAUDE.md (nome novo e explícito, nunca reaproveitado).
