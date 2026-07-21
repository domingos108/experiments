# Análise aprofundada — `chamados_v4_fs_{ftest,mutualinfo,rfembedded,lasso}` (Tarefa 3.7)

Investigação sobre artefatos já gerados (`.pkl` + CSVs de `data/result/`/`results/`), sem
nenhuma nova execução de modelo. Todos os números abaixo foram extraídos diretamente dos
dados reais — reconstruções de `X_train`/`y_train` (quando necessário) usam as mesmas funções
de produção (`input_linear_info` + `fit_predict_model`) e foram validadas comparando o
resultado reconstruído contra o valor persistido no `.pkl` antes de qualquer conclusão.

Formato escolhido: documento Markdown (não notebook). Justificativa: o objetivo é responder
perguntas pontuais com evidência já existente — não há pipeline a re-executar de forma
repetível, e um `.md` é mais direto para colar como rascunho de seção de resultados da
dissertação do que um notebook que precisaria ser "rodado" para ter valor.

---

## Pergunta 1 — `airlines`: o baseline é genuinamente ótimo, ou o Grid Search errou o `k`?

Curva completa de erro de validação vs. `k`, extraída via `load_grid_search_history()`
(Tarefa 3.4) dos `.pkl` reais de `chamados_v4_fs_ftest` e `chamados_v4_fs_mutualinfo`
(3 valores de `estimator__hidden_layer_sizes` × 5 valores de `k`):

**f_test / airlines** (`val_metric_mean`, RMSE de validação):

| hidden_layer_sizes | k=1 | k=5 | k=9 | k=15 | k=20 |
|---|---|---|---|---|---|
| 10 | 14.992 | **14.315** | 14.856 | 14.819 (σ=2.25) | 16.784 (σ=2.43) |
| 20 | 14.992 | **14.315** | 14.851 | 14.419 | 14.840 |
| 50 | 14.993 | **14.314** | 14.883 | 14.363 | 14.719 |

**mutual_info / airlines**:

| hidden_layer_sizes | k=1 | k=5 | k=9 | k=15 | k=20 |
|---|---|---|---|---|---|
| 10 | 14.674 | 14.415 | **14.068** | 14.100 | 15.683 (σ=3.76) |
| 20 | 14.657 | 14.413 | **14.068** | 14.648 (σ=1.51) | 14.776 |
| 50 | 14.748 | 14.384 | **14.047** | 14.064 | 14.730 |

**Resultado: nem monotônico, nem "k errado".** Existe um mínimo claro em `k` intermediário
(k=5 para f_test, k=9 para mutual_info) — `k=20` (sem redução alguma) é consistentemente o
**pior ou quase pior** ponto da curva, inclusive com desvio-padrão disparado entre repetições
(até σ=3.76 para mutual_info k=20/hls=10 — instabilidade real, não só erro maior). Média de
`k=20` em f_test: 15.45; em k=5: 14.31. Média de `k=20` em mutual_info: 15.06; em k=9: 14.06.

Além disso, o `k` escolhido pelo GridSearch nos `.pkl` reais bate exatamente com o mínimo desta
curva: `results/chamados_v4_fs_comparison.csv` mostra `ftest_NFeatures=5` e
`mutualinfo_NFeatures=9` — ou seja, **o Grid Search convergiu corretamente para o próprio
ótimo de validação**, não escolheu um `k` subótimo por acidente.

**Conclusão:** a hipótese "reduzir dimensionalidade em si prejudica `airlines`" não se sustenta
na forma monotônica sugerida pela análise preliminar — há sim uma vantagem de validação em
reduzir features (de k=20 para k=5/9). O que falha é a generalização: mesmo no seu próprio
ótimo de validação, o modelo com FS (RMSE de teste 19.11 para f_test) ainda perde para o
baseline real sem Pipeline/seletor (RMSE de teste 17.27 — `results/chamados_v4_fs_comparison.csv`).
Isso é uma lacuna validação↔teste, não um problema de escolha de `k` — ver Pergunta 4 para um
fator estrutural que ajuda a explicar essa lacuna especificamente em `airlines`.

---

## Pergunta 2 — Tabela de consenso de lags entre os 4 métodos

Metodologia (decisão de design não 100% explícita no pedido original — reportando aqui):
como `rf_embedded` varia o conjunto selecionado entre repetições (ver Pergunta 3), a "moda"
de um conjunto inteiro nem sempre é representativa — em `airlines`/rf_embedded, o conjunto
mais frequente ocorreu em só 2 de 10 repetições (20%). Em vez disso, um método "vota" num lag
se esse lag apareceu em **≥50% das 10 repetições** daquele método (frequência de seleção por
lag, não por conjunto exato). Para os 3 métodos determinísticos (f_test, mutual_info, lasso)
isso equivale exatamente à seleção única e estável (0% ou 100% sempre). `*` marca voto.

**airlines** (20 lags candidatos):

| lag | f_test | mutual_info | rf_embedded | lasso | consenso |
|---|---|---|---|---|---|
| lag_20 | 100%\* | 0% | 100%\* | 100%\* | **3** |
| lag_2 | 100%\* | 100%\* | 100%\* | 0% | **3** |
| lag_1 | 100%\* | 100%\* | 0% | 0% | 2 |
| lag_18 | 0% | 100%\* | 80%\* | 0% | 2 |
| lag_13 | 100%\* | 0% | 80%\* | 0% | 2 |
| lag_12 | 0% | 100%\* | 100%\* | 0% | 2 |
| lag_4 | 0% | 100%\* | 50%\* | 0% | 2 |
| lag_16 | 100%\* | 0% | 30% | 0% | 1 |
| lag_19/17/14/5/6 | (variados, ≤1 método) | | | | ≤1 |

**austres** (1 lag candidato — caso degenerado):

| lag | f_test | mutual_info | rf_embedded | lasso | consenso |
|---|---|---|---|---|---|
| lag_1 | 100%\* | 100%\* | 100%\* | 100%\* | **4** |

**coloradoRiver** (16 lags candidatos):

| lag | f_test | mutual_info | rf_embedded | lasso | consenso |
|---|---|---|---|---|---|
| lag_11 | 100%\* | 100%\* | 100%\* | 100%\* | **4** |
| lag_13 | 100%\* | 100%\* | 100%\* | 0% | **3** |
| lag_12 | 0% | 100%\* | 100%\* | 0% | 2 |
| lag_16/10/9/8/6/3/2 | 100%\* | 100%\* | 0% | 0% | 2 (cada) |
| lag_1 | 0% | 100%\* | 100%\* | 0% | 2 |
| lag_15/7/5/4 | 0% | 100%\* | 0% | 0% | 1 (cada) |

**sunspot** (9 lags candidatos):

| lag | f_test | mutual_info | rf_embedded | lasso | consenso |
|---|---|---|---|---|---|
| lag_9 | 100%\* | 100%\* | 100%\* | 100%\* | **4** |
| lag_4 | 100%\* | 100%\* | 90%\* | 100%\* | **4** |
| lag_8/5/3 | 100%\* | 100%\* | 0% | 0% | 2 (cada) |
| lag_2/1 | 0% | 100%\* | 100%\* | 0% | 2 (cada) |
| lag_7/6 | 0% | 100%\* | 0% | 0% | 1 (cada) |

Lags com consenso ≥3/4 (antes de aplicar o ajuste da Pergunta 5): `airlines` (lag_20, lag_2),
`austres` (lag_1 — trivial, ver nota), `coloradoRiver` (lag_11, lag_13), `sunspot` (lag_9, lag_4).

---

## Pergunta 3 — Variância zero do `rf_embedded` em `coloradoRiver`: sinal dominante, não determinismo oculto

Reconstruí `X_train`/`y_train` do resíduo de `coloradoRiver` (mesma técnica da Tarefa 3.6,
validada contra o `.pkl` real) e refitei `RandomForestRegressor(random_state=None)` **30 vezes
independentes** (fora do wrapper, para inspecionar `feature_importances_` cru, que não
sobrevive no `.pkl` — mesma limitação da Tarefa 3.6).

**Resultado direto: as 30 refits, todas com `random_state=None`, selecionaram exatamente o
mesmo conjunto — `{lag_13, lag_12, lag_11, lag_1}` — 30/30 vezes.** Isso bate com as 10/10
repetições reais persistidas no `.pkl` (`[3, 4, 5, 15]` → mesmos 4 lags).

Importância média ± desvio-padrão por feature entre os 30 refits (evidência de que a
aleatoriedade É real — os desvios-padrão não são zero):

| lag | importância média | desvio-padrão | selecionada? |
|---|---|---|---|
| lag_11 | 0.15945 | 0.00584 | sim |
| lag_12 | 0.14863 | 0.00551 | sim |
| lag_1 | 0.12563 | 0.00438 | sim |
| lag_13 | 0.11795 | 0.00660 | sim (menor das selecionadas) |
| lag_2 | 0.05523 | 0.00325 | não (maior das não-selecionadas) |
| ... (demais 11 lags) | 0.023–0.052 | 0.002–0.004 | não |

**Margem de segurança:** a selecionada mais fraca (lag_13, 0.118) tem importância **~2,1x
maior** que a não-selecionada mais forte (lag_2, 0.055) — um gap de 0.063, equivalente a
cerca de 10 desvios-padrão de qualquer uma das duas. Ou seja, o ruído de amostragem do
Random Forest (σ≈0.003–0.007) é ordens de grandeza pequeno demais para jamais cruzar esse
gap.

**Conclusão: hipótese (a) confirmada, (b) refutada com evidência direta.** É sinal
genuinamente dominante — os 4 lags relevantes (13, 12, 11, 1 — os 3 lags sazonais mais
próximos mais o lag imediato) se destacam tanto que sobrevivem a qualquer reamostragem do RF.
Não há `random_state` fixado/vazando em lugar nenhum: inspecionei o `model_parameters` do
notebook `arima_mlp_rf_embedded.ipynb` — não define `selector__random_state`, então o
default `None` da classe (decisão da Tarefa 3.1) permanece intacto, e os desvios-padrão
não-zero medidos acima confirmam empiricamente que a aleatoriedade está de fato ativa.

**Achado colateral (fora do escopo desta pergunta, mas relevante para uma limpeza futura):**
o `model_parameters` de `arima_mlp_rf_embedded.ipynb` ainda contém
`'selector__k': [1, 5, 9, 15, 20]` — um parâmetro morto, já que `rf_embedded` nunca lê `k`
(comportamento testado em `test_k_parameter_is_ignored_for_this_strategy`). Isso não afeta a
correção do resultado (confirmado agora com a reconstrução independente acima), só infla o
grid search em 5x por parâmetros sem efeito. Reportando, não corrigindo — fora do escopo desta
tarefa.

---

## Pergunta 4 — Razão amostra:features vs. onde FS ajuda ou atrapalha

| Série | `df_train` | `N_Features_Total` | Razão | Resultado (métodos que venceram o baseline, de 4) |
|---|---|---|---|---|
| airlines | 80 | 20 | **4.0 : 1** | 0/4 — perde em todos |
| austres | 71 | 1 | 71.0 : 1 (trivial — só 1 candidato) | 0/4 — perde em todos |
| sunspot | 216 | 9 | 24.0 : 1 | 2/4 — misto |
| coloradoRiver | 568 | 16 | 35.5 : 1 | 4/4 — vence em todos |

(`PctGain` fonte: `results/chamados_v4_fs_comparison.csv` — positivo = FS venceu o baseline.)

**Padrão observado (descritivo, sem teste de hipótese formal — Seção 3.4 do CLAUDE.md):**
excluindo `austres` (caso degenerado, discutido abaixo), há uma relação visível entre a razão
amostra:features e o resultado: `airlines` tem de longe a menor razão (4:1) e é a única série
onde a FS perde nos 4 métodos simultaneamente; `coloradoRiver` tem a maior razão útil (35.5:1)
e vence nos 4; `sunspot` fica no meio (24:1) e o resultado é misto (2 vencem, 2 perdem). Isso é
consistente com uma leitura de que `airlines` tem `df_train` pequeno (80 linhas) demais para
sustentar 20 lags candidatos alimentando uma MLP — reforça, com um mecanismo estrutural
concreto, a conclusão da Pergunta 1 (o problema não é "qual k", é que mesmo o melhor k ainda
opera sob uma razão amostra:features desfavorável comparada às outras 3 séries).

`austres` é um outlier que não se encaixa no padrão por um motivo estrutural diferente:
`N_Features_Total=1` (já documentado na Tarefa 3.2 — `lag_size='auto'` resolve para 1 único
lag via PACF) torna a "seleção de features" um não-evento — não há nada para selecionar. A
razão 71:1 é matematicamente alta mas não é comparável às demais linhas desta tabela.

---

## Pergunta 5 — Integrando o achado do fallback do Lasso (Tarefa 3.6) na Pergunta 2

A Tarefa 3.6 **já foi concluída** nesta mesma sessão, antes desta tarefa. Achado direto
(reconstrução validada contra o `.pkl` real): em `chamados_v4_fs_lasso`, o Lasso caiu no
fallback determinístico de zero-features (Tarefa 3.1) em **3 das 4 séries** — `airlines`,
`austres` e `coloradoRiver` — e só selecionou genuinamente (sem fallback) em `sunspot`.

Impacto direto na leitura da tabela de consenso da Pergunta 2:

- **`airlines` / lag_20 (consenso "3/4"):** o voto do lasso aqui **não é seleção genuína** —
  é o argmax de coeficientes todos ~0 (fallback). O consenso real e independente para lag_20
  é **2/4** (f_test + rf_embedded), no mesmo patamar de várias outras lags da tabela — não se
  destaca como "consenso forte" depois de descontar o artefato.
- **`airlines` / lag_2 (consenso "3/4", sem voto do lasso):** aqui os 3 votos (f_test,
  mutual_info, rf_embedded) já eram todos genuínos — **esta é a evidência de consenso mais
  confiável em `airlines`**, não lag_20.
- **`coloradoRiver` / lag_11 (consenso "4/4"):** o voto do lasso também é fallback aqui — mas
  isso **não enfraquece** a conclusão, porque os outros 3 métodos (f_test, mutual_info,
  rf_embedded) já convergiram genuinamente e independentemente no mesmo lag (e a Pergunta 3
  mostrou que o sinal do rf_embedded para lag_11 é fortíssimo, gap de ~10σ). O consenso
  genuíno real é 3/4 — ainda o mais forte da série, só não "unânime" de fato.
- **`austres` / lag_1 (consenso "4/4"):** duplamente inflado — além do lasso ter caído em
  fallback (mesmo com um único candidato, o coeficiente zerou), `N_Features_Total=1` (Pergunta
  4) já tornava a "seleção" trivial para os outros 3 métodos também. Esta linha não deve ser
  lida como evidência de relevância de `lag_1` — é um artefato estrutural somado a um
  artefato de fallback.
- **`sunspot` / lag_9 e lag_4 (consenso "4/4"):** o Lasso **não** caiu em fallback aqui
  (Tarefa 3.6 confirmou seleção genuína, 2 features via corte normal do `SelectFromModel`) —
  este é o único caso de consenso 4/4 **genuíno e sem ressalvas** entre as 4 séries.

**Ranking final de confiabilidade da evidência de consenso, do mais ao menos forte:**
1. `sunspot` lag_9/lag_4 — 4/4 genuíno.
2. `coloradoRiver` lag_11 — 3/4 genuíno (+ margem de sinal enorme via Pergunta 3).
3. `airlines` lag_2 — 3/4 genuíno.
4. `coloradoRiver` lag_13 — 3/4 genuíno (não envolve o lasso).
5. `austres` lag_1 e `airlines` lag_20 — não devem ser citados como "consenso forte" sem a
   ressalva de que são inflados por fallback/trivialidade.

---

## Perguntas em aberto (reveladas por esta investigação, sem resposta ainda)

1. Por que o modelo com FS, mesmo no seu próprio ótimo de validação, ainda perde para o
   baseline real em `airlines`? A razão amostra:features (Pergunta 4) é um fator estrutural
   plausível, mas não foi isolado de outras diferenças entre `arima_mlp.ipynb` (baseline) e
   `arima_mlp_ftest.ipynb` (ex.: grid de hiperparâmetros da MLP pode diferir entre os dois
   notebooks — não verificado aqui, fora do escopo desta tarefa).
2. `austres` deveria continuar sendo incluída nas comparações agregadas de FS? Já era uma
   dúvida aberta desde a Tarefa 3.2 (comentário no próprio notebook), e esta tarefa reforça
   que ela distorce qualquer leitura agregada (ratio, consenso) por ter só 1 feature candidata.
3. O `selector__k` morto em `arima_mlp_rf_embedded.ipynb` (achado colateral da Pergunta 3) —
   provavelmente existe o mesmo resíduo nos notebooks `mutualinfo`/`lasso`? Não verificado
   aqui (só confirmei o de `rf_embedded`, que foi o relevante para a Pergunta 3).
4. `sunspot` rf_embedded/lasso ganham +0.48%/+0.47% do baseline — a própria tarefa já
   qualificou isso como "inconclusivo, margem da ordem do desvio-padrão entre repetições";
   esta investigação não teve escopo para aprofundar essa margem especificamente (seria a
   Pergunta 6, não pedida aqui).

---

## Ambiguidades / decisões de design não 100% explícitas no pedido original

- **Definição operacional de "seleção mais frequente/estável"** (Pergunta 2): usei frequência
  de seleção por lag ≥50% entre as 10 repetições de cada método, em vez de "moda do conjunto
  exato" — porque em `airlines`/`rf_embedded` o conjunto mais frequente ocorreu em só 2/10
  repetições, uma base fraca demais para representar "a escolha estável do método". Detalhado
  na seção da Pergunta 2.
- **Reconstrução de dados para a Pergunta 3**: segui exatamente a técnica já usada e validada
  na Tarefa 3.6 (mesmas funções de produção, mesma validação cruzada contra o `.pkl` real
  antes de tirar qualquer conclusão) — não inventei um método novo.
- Não toquei no achado colateral do `selector__k` morto em `rf_embedded` — reportado como
  pergunta em aberto, não como correção, por estar fora do escopo desta tarefa (que é somente
  leitura/análise).

---

## Adendo (Tarefa 3.8) — fechamento das 2 perguntas em aberto acima

### Grid de MLP: idêntico entre baseline e os 4 notebooks FS — MAS o `.pkl` do baseline está desatualizado

O grid de hiperparâmetros da MLP (`hidden_layer_sizes=[10,20,50]`, `max_iter=[1000]`,
`activation='logistic'`, `solver='lbfgs'`, `model_exec=10`, `use_val_slipt_for_prev=True`
efetivo) é **idêntico**, célula a célula, entre `arima_mlp.ipynb` (baseline) e os 4 notebooks
`arima_mlp_<estrategia>.ipynb` — confirmado lendo o JSON bruto dos 5 notebooks e o código-fonte
de `grid_seach_multiple_bases` (default `use_val_slipt_for_prev=True`, igual ao valor explícito
passado pelos 4 notebooks FS).

**Porém — achado real, não hipotético:** o `.pkl` do baseline atualmente em
`data/result/chamados/*_1amv1.pkl` (usado em TODAS as comparações desta análise, Tarefas 3.1–3.7)
foi persistido com `MLPRegressor(activation='identity', ...)`, enquanto o código-fonte ATUAL de
`arima_mlp.ipynb` — e todos os 4 `.pkl` de FS — usam `activation='logistic'`. Confirmado nas 4
séries (`airlines`, `austres`, `coloradoRiver`, `sunspot`) inspecionando o objeto `MLPRegressor`
fitted dentro de cada `.pkl`. `git log -p` em `arima_mlp.ipynb` mostra que o commit `43dae50`
(2026-07-02) alterou `activation` de `'identity'` para `'logistic'`; o `.pkl` do baseline tem
mtime de 2026-03-27 — **anterior** a esse commit. Como `*.pkl` está no `.gitignore` e o notebook
roda com `force=False`, o `.pkl` nunca foi regenerado após a mudança — exatamente o cenário de
"idempotência silenciosa" que a Seção 3.2 do CLAUDE.md documenta como intencional, mas cuja
responsabilidade de evitar confusão é do pesquisador via disciplina manual.

**Impacto:** `identity` (ativação linear na camada oculta) e `logistic` (sigmoide, não-linear)
são arquiteturas com poder expressivo fundamentalmente diferentes. Isso significa que **todas as
comparações "FS vs. baseline" feitas até agora (Tarefas 3.1–3.7) estão comparando os 4 métodos
de FS (rodados com `logistic`) contra um baseline que não reflete mais o próprio código-fonte
atual do projeto** — não é um problema de grid de hiperparâmetros (esses batem), é um artefato
persistido desatualizado. Isso qualifica — não necessariamente invalida — as conclusões: o
sinal mais forte da Pergunta 4 (razão amostra:features explicando por que `airlines` perde) pode
estar parcialmente confundido com essa diferença de ativação, que não foi isolada.

### `selector__k` morto: presente em `rf_embedded` E `lasso`, não em `ftest`/`mutual_info`

Confirmado lendo os 4 notebooks: `ftest` e `mutual_info` usam `selector__k` legitimamente (são
os 2 métodos baseados em `k`). `rf_embedded` e `lasso` também têm `'selector__k': [1, 5, 9, 15, 20]`
no dicionário `model_parameters` — **apesar do comentário na própria célula dizer "Tarefa 3.1:
selector__k REMOVIDO"** — ou seja, o comentário está certo sobre a intenção, mas o código não
foi atualizado para refletir isso nos 2 notebooks embedded.

Reprodução mínima com dados sintéticos (sem tocar em nenhum `.pkl` real):
1. `Pipeline(...).set_params(selector__k=999999)` para `strategy='rf_embedded'` **não levanta
   erro** — `k` é um parâmetro real do construtor de `TimeSeriesFeatureSelector` (existe para
   as 4 estratégias, só não é lido por 2 delas), então o sklearn aceita e seta silenciosamente.
2. `selected_indices_` para `k=1` e `k=999999` são **idênticos** — confirma que não há efeito
   funcional na seleção (já coberto por teste unitário existente,
   `test_k_parameter_is_ignored_for_this_strategy`, mas reproduzido aqui de forma isolada).
3. Com um estimador determinístico no lugar da MLP, as combinações "duplicadas" do
   `ParameterGrid` (`k=1`, `k=5`, `k=9`, mesmo `hidden_layer_sizes`) produzem `MSE` **byte-
   idêntico** — prova mecânica de que o parâmetro é 100% inerte fim-a-fim, não só na etapa do
   seletor.
4. Confirmado por grep: nenhum dos 5 notebooks passa `random_state` ao `MLPRegressor` — na
   produção real (ao contrário do repro determinístico acima), as 5 "cópias" de cada
   `hidden_layer_sizes` no grid de `rf_embedded`/`lasso` produzem `val_metric_mean`
   **ligeiramente diferentes só por ruído de inicialização da MLP**, não por qualquer efeito
   genuíno do seletor.

**Consequência real (não hipotética) sobre o resultado persistido:** o parâmetro morto NÃO
alterou quais features foram selecionadas (item 2 acima) nem introduziu nenhum erro silencioso
na seleção em si. O efeito real é (a) inflar o grid search em 5x (15 combinações ao
invés de 3, cada uma com `model_exec=10` repetições internas — 150 fits de MLP em vez de 30,
só para a fase de busca) e (b) um viés otimista leve na escolha de `best_params`: como
`_search_params()` faz `np.argmin` sobre TODAS as 15 combinações (grid_search_exp.py:142),
e 5 dessas 15 são réplicas ruidosas-mas-funcionalmente-idênticas de cada `hidden_layer_sizes`,
o mínimo escolhido tende a ser o mais "sortudo" dentre 5 tentativas em vez de 1 — um efeito de
comparações múltiplas real, porém pequeno (RMSE de validação, não de teste; a métrica final
reportada usa 10 repetições do refit final com os `best_params` já escolhidos, então o impacto
fica restrito a QUAL configuração de `hidden_layer_sizes` foi escolhida como "melhor", não ao
valor de RMSE final reportado em si).

### Recomendação

- **Baseline (`activation`): recomendo fortemente re-rodar `arima_mlp.ipynb` com `force=True`**
  antes de qualquer texto final de resultados — é a única forma de eliminar a divergência
  `identity` vs. `logistic`, que hoje contamina TODAS as comparações FS-vs-baseline já feitas.
  Isso toca um dos 5 `.pkl` protegidos pela Seção 3 do CLAUDE.md — não fiz isso por conta
  própria; aguardando sua decisão.
- **`selector__k` morto em `rf_embedded`/`lasso`:** não corrigi (fora do escopo desta tarefa).
  Recomendo remover a chave dos 2 notebooks — mas como isso muda o grid search real (15→3
  combinações), regenerar os `.pkl` de `chamados_v4_fs_rfembedded`/`chamados_v4_fs_lasso`
  depois seria uma boa prática, não estritamente obrigatório (o resultado de seleção de
  features já é comprovadamente idêntico com ou sem essa chave — a única mudança seria
  eliminar o viés leve de múltiplas comparações na escolha de `hidden_layer_sizes` e economizar
  tempo de execução).
- Os resultados de **consenso de lags** (Pergunta 2/3.7) e a investigação de **variância do RF**
  (Pergunta 3/3.7) continuam válidos como estão — nenhum dos dois achados desta tarefa afeta a
  seleção de features em si, só a comparação de métrica final contra o baseline.
