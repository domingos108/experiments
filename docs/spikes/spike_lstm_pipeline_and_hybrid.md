# Spike — `CustomLSTM` em `Pipeline`: single-model (reconfirmação) e híbrido ARIMA-LSTM

Investigação de viabilidade, sem implementação de notebook/`.pkl` de produção. Cobre 3
perguntas: (A) `CustomLSTM` + `Pipeline([TimeSeriesFeatureSelector, CustomLSTM])` ainda
encaixa como single-model agora que existe o mecanismo de seed
(`estimator_random_state_base`, CLAUDE.md 3.4)? (B) a mesma composição funciona dentro de
`Additive` (híbrido ARIMA-LSTM), nunca testado antes? (C) a ressalva conceitual de lags
não-contíguos continua valendo à luz dos dados reais já gerados pela matriz `auto`/`pct10`?

**Resposta curta:** (A) sim, mecanicamente — mas a seed NÃO alcança `CustomLSTM` (falha alto,
não silenciosamente). (B) sim, sem nenhuma mudança de código. (C) sim, a ressalva continua
válida — nenhum dos 5 métodos produz seleções contíguas por padrão; `rfecv` é o menos ruim,
`rf_embedded` é o pior, mas por razões que precisam de leitura cuidadosa (ver Parte C).

---

## Parte A — Single-model (`SKlearnModel`): reconfirmação + seed

`CustomLSTM` (`src/model/lstm.py`) é um `BaseEstimator`/`RegressorMixin` real sobre um
`Sequential([Input, LSTM, Dense])` do Keras — não é um wrapper `model_class_exp` tipo
LSTM/NHITS/ELM/SCN (`neural_forecast_exp.py`/`perturbative_neural_forecast.py`), que usam
`random_seed` próprio e passam por um caminho de `GridSearch` inteiramente diferente
(PLANO_ARQUITETURA.md 1.6, "bug real corrigido... `model_class_exp` não-sklearn"). Isso
importa porque `is_not_sklearn(model) = not isinstance(model, BaseEstimator)` — `CustomLSTM`
passa nesse teste como sklearn de verdade, então trafega pelo mesmo caminho de código já
validado para MLP/SVR (`fit_predict_ml_schemma`, `clone(...).set_params(...)`), não pelo
caminho `model_class_exp`.

**Teste real** (`tests/model/test_single_ml_model_exp.py::TestSKlearnModelAcceptsPipelineWithLSTM`,
dado real de `airlines.txt`, `epochs=2`/`hidden_layer_sizes=5` propositalmente mínimos —
o objetivo é confirmar que a mecânica de reshape 2D→3D dentro de `CustomLSTM.fit`/`predict`
encaixa no contrato `model.fit(x_train, y_train)`/`model.predict(x)`, não que o resultado
seja bom):

```python
model = Pipeline([
    ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
    ("estimator", CustomLSTM(hidden_layer_sizes=5, epochs=2)),
])
# ... GridSearch(single_ml_model_exp.SKlearnModel, model, ...).execution()
```

Resultado: roda de ponta a ponta sem nenhuma mudança em `single_ml_model_exp.py`/
`grid_search_exp.py`. `n_features_in_ == 20` (mesmo `lag_size='auto'` de sempre para
`airlines`), `selected_indices_ == [7, 8, 19]` (já não-contíguo — ver Parte C).
`test_metrics` não veio vazio, mas o RMSE de teste (~500, contra ~20-30 de MLP/SVR no mesmo
setup) é visivelmente ruim — esperado com 2 épocas, **não é uma medida de qualidade real do
LSTM**, só confirma que o pipeline mecanicamente produz uma previsão.

### A seed (`estimator_random_state_base`) NÃO alcança `CustomLSTM`

`CustomLSTM.__init__(self, hidden_layer_sizes=50, epochs=1000)` não declara `random_state` —
`Pipeline([...]).get_params(deep=True)` confirma (evidência real, não suposição):

```
estimator, estimator__epochs, estimator__hidden_layer_sizes, memory,
selector, selector__k, selector__random_state, selector__strategy, steps,
transform_input, verbose
```

Sem `estimator__random_state`. `_clone_model_for_rep` (`grid_search_exp.py:136-170`) checa,
nesta ordem: `estimator__random_state` no Pipeline → `random_state` no estimador nu → se
nada disso existir E o model for um Pipeline (`hasattr(model_actual, "named_steps")`) →
**`ValueError`** (falha alto, mesmo comportamento já garantido pelo teste existente
`test_seed_on_pipeline_without_estimator_random_state_raises`, que cobre o caso de step mal
nomeado — este é um caso distinto: o step **se chama** `estimator` corretamente, só que o
próprio estimador não tem `random_state` nenhum).

**Achado extra, não previsto no plano original:** o teste de guarda em `GridSearch.__init__`
(`is_not_sklearn(model) and estimator_random_state_base is not None → raise`) **não pega esse
caso** — como `CustomLSTM` é um `BaseEstimator` de verdade, `is_not_sklearn` retorna `False`
e a construção do `GridSearch` sucede silenciosamente. A falha só aparece depois, dentro de
`execution()`, na primeira chamada a `_clone_model_for_rep` (primeira repetição da primeira
combinação do grid) — ainda falha alto e cedo (antes de qualquer treino real rodar), mas não
no ponto mais óbvio (construção do objeto). Confirmado com teste real:
`tests/model/test_grid_search_seed.py::TestSeedParamWiring::test_seed_with_customlstm_estimator_raises_no_random_state_param`.

**Conclusão A:** `CustomLSTM` single-model continua encaixando mecanicamente, sem mudança de
código. Mas **não há reprodutibilidade via `estimator_random_state_base`** — para semear
`CustomLSTM` seria necessário um mecanismo próprio (`tf.random.set_seed(...)`/seed interna do
Keras, hoje inexistente na classe), não a convenção sklearn já usada por MLP. Se uma família
LSTM entrar em produção com `model_exec=10` (mesma convenção estocástica de MLP, CLAUDE.md
3.4), ela rodaria **sem seed nenhuma** com o mecanismo atual — reprodutibilidade entre
execuções ficaria pior que a já obtida para MLP, a menos que se implemente seeding próprio
em `CustomLSTM` antes (fora do escopo desta investigação).

---

## Parte B — Híbrido ARIMA-LSTM (`Additive`): não testado antes

`Additive.fit_predict()` delega a `generics.fit_predict_model()` → `fit_predict_ml_schemma()`
— a mesma função 100% agnóstica (`model.fit(x_train, y_train)`/`model.predict(x)`) já
comprovada para MLP (Tarefas 2/3), SVR (`TestAdditiveAcceptsPipelineWithSVR`) e
`KhasheiBijariHybrid`. Não havia razão estrutural a priori para `CustomLSTM` se comportar
diferente aqui — mas, seguindo a mesma disciplina já aplicada a
`KhasheiBijariHybrid`/`NoLiCHybrid` (não presumir compatibilidade sem testar), foi testado
com dado real.

**Teste real** (`tests/model/test_hybrid_system_exp.py::TestAdditiveAcceptsPipelineWithLSTM`,
`.pkl` real do ARIMA de `airlines.txt` copiado para um `MODEL_DATA_PATH` temporário, mesmo
padrão de `TestAdditiveAcceptsPipelineWithSVR`):

```python
model = Pipeline([
    ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
    ("estimator", CustomLSTM(hidden_layer_sizes=5, epochs=2)),
])
# ... GridSearch(hybrid_system_exp.Additive, model, ...).execution()
```

Resultado: roda de ponta a ponta **sem nenhuma mudança em `hybrid_system_exp.py`**.
`n_features_in_ == 20` (resíduo do ARIMA, `lag_size='auto'`), `selected_indices_ == [0, 4, 18]`
(não-contíguo — ver Parte C). `test_metrics` não veio vazio, e desta vez o RMSE de teste
(~33.4) é **plausível** (na faixa de MLP/SVR híbridos, ~20-30) mesmo com só 2 épocas — porque
o LSTM aqui só precisa modelar o resíduo do ARIMA (escala pequena), não a série bruta. Isso
não é uma medida de qualidade real (2 épocas ainda é insuficiente para convergir de verdade),
só evidência de que a recomposição $\hat{L}_t + \hat{N}_t$ funciona mecanicamente igual ao
caso MLP/SVR.

**Confirmação específica pedida na tarefa — cópia do ARIMA + recomposição:** ambas funcionam
sem alteração. A cópia de `chamados/airlines_1arima.pkl` para dentro do `experiment_id` fake
(via `utils.copy_pretrained_linear_model`, mesmo mecanismo usado por todo `Additive`) não
precisou de nenhum tratamento especial para `CustomLSTM` — `Additive` não distingue a
identidade do estimador não-linear ao carregar o modelo linear pré-treinado. A inversão de
normalização/diferenciação de $y$ (MinMaxScaler + KPSS, fora do `Pipeline` por design,
PLANO_ARQUITETURA.md 1.4) também não precisou de mudança — `test_metrics` populado confirma
que a cadeia completa (`Additive.fit_predict` → `generics.format_forecats`) rodou.

**Conclusão B:** `Additive` + `Pipeline([selector, CustomLSTM])` encaixa **sem nenhuma
mudança estrutural** — mesmo veredito de `KhasheiBijariHybrid`, e pela mesma razão raiz
(agnosticismo de `fit_predict_ml_schemma`). A mesma ressalva de seed da Parte A se aplica
aqui identicamente (não testado separadamente porque `_clone_model_for_rep` é compartilhado
por `Additive`/`SKlearnModel` — a Parte A já cobre o mecanismo).

---

## Parte C — Reavaliação da ressalva conceitual (lags não-contíguos)

A ressalva original (spike anterior, não versionado neste repositório — só citado no pedido
desta tarefa): `CustomLSTM.fit()` faz `X.reshape((n, tw_size, 1))`, tratando cada coluna
selecionada como um passo de tempo sequencial. Se o seletor devolve um subconjunto
não-contíguo de lags (ex. `lag_20, lag_15, lag_3`), o LSTM recebe uma sequência de 3 passos
com espaçamento temporal real de 5 e 12 unidades, mas trata como se fossem 3 passos
consecutivos — quebrando a premissa de regularidade que justifica LSTM sobre MLP.

**Isso não é uma falha de execução** (nenhum dos dois testes acima lançou erro por causa da
não-contiguidade) — é um problema silencioso de adequação estatística/semântica, exatamente o
tipo de risco que `code-review`/`systematic-debugging` não pegam sozinhos porque o código
"funciona".

### Dado real: os 5 métodos produzem seleção contígua com que frequência?

Extraído de **todos** os `selected_features_detail.csv` já existentes nas matrizes `auto` e
`pct10` (20 diretórios de FS × 2 janelas = 40 arquivos, 1320 linhas de repetição, 985 com
`N_Features_Selected > 1` — linhas triviais de 1 feature, como `austres`/`auto`, foram
excluídas por não terem contiguidade a medir). Duas métricas por linha, sobre os índices
posicionais `Selected_Indices`:

- **`density`** = `n_selecionadas / (max(índice) − min(índice) + 1)` — 1.0 = bloco
  perfeitamente contíguo; quanto menor, mais espalhado dentro do intervalo que ocupa.
- **`adjacency_ratio`** = fração de pares consecutivos na lista ordenada de índices que são
  vizinhos (`idx[i+1] - idx[i] == 1`).

| Estratégia | n linhas válidas | density média | adjacency_ratio média | nº médio de features selecionadas |
|---|---|---|---|---|
| **rfecv** | 229 | **0.732** | **0.689** | 19.93 |
| mutual_info | 211 | 0.610 | 0.662 | 11.51 |
| lasso | 122 | 0.573 | 0.531 | 8.89 |
| f_test | 209 | 0.556 | 0.573 | 9.36 |
| **rf_embedded** | 214 | **0.378** | **0.298** | 9.14 |

**Achado central: nenhum método produz seleção majoritariamente contígua** (`density` bem
abaixo de 1.0 em todos os 5) — a ressalva conceitual continua válida de forma geral, não é
um problema teórico que a prática dissipa.

**Mas há um gradiente real, com uma ressalva importante de leitura:** `rfecv` tem a maior
`density`/`adjacency_ratio` — só que também seleciona, em média, **quase a janela inteira**
(19.93 de ~20, tipicamente eliminando só 1-7 lags). Um método que mantém quase tudo
naturalmente parece "contíguo" (poucos buracos possíveis). A comparação mais limpa é entre
`f_test` (9.36 features em média) e `rf_embedded` (9.14, praticamente o mesmo número) — aqui,
com contagem de features equivalente, `rf_embedded` (density 0.378) é visivelmente mais
espalhado que `f_test` (density 0.556). Isso não é artefato de "selecionar mais ou menos", é
uma diferença real de **como** cada método distribui a escolha dentro da janela.

**Contra-exemplo real que qualifica até `rfecv`:** quando `rfecv` de fato poda agressivamente
(não o caso típico), o resultado pode ser tão ou mais espalhado que os outros métodos —
`chamados_pct10_fs_mlp_rfecv/coloradoRiver` (repetição 3): `lag_60, lag_48, lag_36, lag_24,
lag_12, lag_1` — 6 features, density=0.10, um padrão quase perfeitamente harmônico (múltiplos
de 12) que faz sentido para uma série com sazonalidade mensal, mas é o oposto de "contíguo".

**Não há separação limpa por janela** (`auto` vs. `pct10`) — as médias de `density` por
método são parecidas nas duas janelas (ex. `rfecv`: 0.752 auto vs. 0.715 pct10; `rf_embedded`:
0.369 auto vs. 0.385 pct10), sugerindo que o padrão de contiguidade é uma propriedade do
método de seleção, não do tamanho da janela candidata.

**Conclusão C:** a ressalva conceitual **continua sendo um problema real e geral** — não há
método que "resolva" a questão produzindo seleções tipicamente contíguas. Dito isso, se um
dia LSTM entrar em produção, **`rfecv` é o candidato menos prejudicial por padrão** (maior
density média, embora parcialmente por reter quase todas as features) e **`rf_embedded` é o
mais prejudicial** (menor density mesmo com contagem de features comparável a `f_test`/
`lasso`) — uma diferença mensurável, não hipotética, mas que não elimina o problema para
nenhum dos 5 métodos.

---

## Recomendação

**Viável implementar os dois (single-model e híbrido)**, do ponto de vista estrutural — nenhum
dos dois precisa de mudança em `single_ml_model_exp.py`, `hybrid_system_exp.py` ou
`grid_search_exp.py`, mesmo padrão "zero mudança" já estabelecido para MLP/SVR/
KhasheiBijariHybrid. Duas condições precisam ser resolvidas ANTES de qualquer notebook de
produção, não depois:

1. **Reprodutibilidade:** se a família LSTM for rodar com `model_exec > 1` (convenção
   estocástica de MLP), o mecanismo `estimator_random_state_base` atual não a alcança — ou se
   aceita rodar sem seed (mesma situação de MLP antes da Tarefa correspondente), ou se
   implementa seeding próprio em `CustomLSTM` (`tf.random.set_seed` dentro de `fit()`,
   parametrizado) antes.
2. **Ressalva de contiguidade:** dado que nenhum método a resolve, decidir com o
   orientador se isso invalida o uso de LSTM como estimador de FS nesta fase, ou se é aceito
   como limitação documentada (ex.: reportar como trabalho futuro "ordenar/reindexar lags
   selecionados antes do reshape" — não implementado nem desenhado aqui, fora de escopo desta
   investigação).

Nenhuma decisão de escopo foi tomada aqui — só a viabilidade estrutural (sim) e o estado real
dos dois problemas conhecidos (seed não alcança; contiguidade não é resolvida por nenhum
método, com gradiente mensurável entre eles).

## Custo prático não-funcional (achado incidental)

`import tensorflow` sozinho levou ~15-20s neste ambiente (primeira importação por processo).
`CustomLSTM.__init__` chama `K.clear_session()` a cada instanciação, e o módulo `lstm.py` cria
uma sessão TF1-compat (`tf.compat.v1.Session`/`ConfigProto`) no nível do módulo, com
`tf.ConfigProto`/`K.set_session` marcados como deprecados pelo próprio TensorFlow (warnings
reais capturados ao rodar os testes, não bloqueantes). Isso não impede o uso, mas é uma
fragilidade a observar: nenhum teste do projeto importava `model.lstm` antes desta
investigação — os 3 testes novos adicionam esse custo de import à suíte pela primeira vez.

## Testes: formalizados como regressão permanente

- `tests/model/test_single_ml_model_exp.py::TestSKlearnModelAcceptsPipelineWithLSTM` — espera
  sucesso end-to-end (Parte A).
- `tests/model/test_hybrid_system_exp.py::TestAdditiveAcceptsPipelineWithLSTM` — espera
  sucesso end-to-end (Parte B).
- `tests/model/test_grid_search_seed.py::TestSeedParamWiring::test_seed_with_customlstm_estimator_raises_no_random_state_param` —
  documenta o comportamento ATUAL (falha alto na primeira repetição, não na construção) via
  `pytest.raises`. Se `CustomLSTM` ganhar `random_state` próprio no futuro, este teste precisa
  ser atualizado para esperar sucesso, não removido silenciosamente.

## Ambiguidades — reportadas, não resolvidas

1. **Spike anterior citado no pedido** (`docs/spikes/spike_lstm_gan_nhits.md` ou resultado
   equivalente que teria "confirmado `CustomLSTM` como wrapper completo/funcional") **não
   existe neste repositório** — não encontrado em `docs/spikes/`, `CHECKPOINTS.md`,
   `CLAUDE.md` ou `PLANO_ARQUITETURA.md`. Esta investigação repartiu do zero (Parte A incluída,
   não só a B), com evidência própria, em vez de presumir o achado citado. Se esse spike
   existiu numa sessão anterior não commitada, seus achados não foram comparados aqui —
   reportar para o pesquisador confirmar se o conteúdo desta Parte A bate com o que já era
   sabido.
2. **Definição de "contiguidade"** (Parte C): usei `density` (razão features/span) e
   `adjacency_ratio` (fração de pares vizinhos) como proxies quantitativos — não são as únicas
   métricas possíveis, e a escolha afeta a magnitude (não a direção) do ranking entre métodos.
   `rfecv` liderar em density está parcialmente confundido com reter mais features (discutido
   acima) — reportado explicitamente para não inflar a conclusão além do que o dado sustenta.
3. **Métrica de qualidade do LSTM não avaliada de propósito** — os 2 testes usam `epochs=2`
   deliberadamente (checar mecânica de encaixe, não desempenho). Nenhuma conclusão sobre
   "LSTM é bom/ruim para esta tarefa" pode ser extraída dos RMSEs reportados aqui.
