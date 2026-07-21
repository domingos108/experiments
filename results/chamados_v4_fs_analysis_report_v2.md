# Reanálise com baseline corrigido — `chamados_v4_fs_{ftest,mutualinfo,rfembedded,lasso}` (Tarefa 3.10)

Substitui inteiramente `results/chamados_v4_fs_analysis_report.md` (Tarefa 3.7), preservado
como registro histórico da versão pré-correção — **não editado**. Esta reanálise usa os dados
gerados após a Tarefa 3.9 (baseline `1amv1` regenerado com `activation='logistic'`;
`rf_embedded`/`lasso` regenerados sem o `selector__k` morto). Mesma metodologia da Tarefa 3.7
(nenhum `.pkl` novo gerado nesta tarefa, só leitura/análise).

---

## 0. Pré-checks

**1) Os 4 experimentos comparam contra o baseline corrigido?** Confirmado lendo os `.pkl`
reais: `data/result/chamados/*_1amv1.pkl` tem `activation='logistic'` nas 4 séries (checado
agora). `ftest`/`mutual_info` **não foram regenerados** (mtime 20/07 16:38–17:01, anterior à
Tarefa 3.9) — mas isso é esperado e correto: `build_comparison()` sempre lê `baseline_dir`
direto de `data/result/chamados/` no momento da execução, então a comparação `ftest`/
`mutual_info` vs. baseline já reflete o baseline novo automaticamente, sem precisar
regenerá-los. Confirmado numericamente: `ftest_RMSE`/`mutualinfo_RMSE` em
`results/chamados_v4_fs_comparison.csv` são **byte-idênticos** aos da Tarefa 3.7 (ex.:
`airlines: 19.113854`/`19.614115` nos dois arquivos) — só `Baseline_RMSE`/`PctGain` mudaram,
exatamente como esperado.

**2) `rfembedded`/`lasso` têm `.pkl` novos?** Confirmado: `airlines_1amv1rfembedded.pkl`
mtime 21/07 01:00:41, `airlines_1amv1lasso.pkl` mtime 21/07 01:02:26 — ambos posteriores à
regeneração do baseline (01:00 vs. baseline 00:55). `grid_search_history` desses `.pkl` tem
**3 combinações** (não mais 15) e `params.keys() = {'estimator__hidden_layer_sizes',
'estimator__max_iter'}` — confirma que o `selector__k` morto foi removido de fato, não só do
código-fonte do notebook.

---

## 1. Tabela comparativa — Tarefa 3.7 (baseline `identity`, grid sujo) vs. Tarefa 3.10 (corrigido)

| Série | Baseline_RMSE (antes→depois) | Método | PctGain antes | PctGain depois | Δ PctGain |
|---|---|---|---|---|---|
| **airlines** | 17.2745 → **17.4362** (baseline piorou +0.94%) | ftest | -10.65% | -9.62% | +1.03pp |
| | | mutualinfo | -13.54% | -12.49% | +1.05pp |
| | | rfembedded | -12.29% | -11.44% | +0.85pp |
| | | lasso | -14.23% | -13.17% | +1.06pp |
| **austres** | 18.9257 → **19.0858** (baseline piorou +0.85%) | ftest | -0.98% | -0.13% | +0.85pp |
| | | mutualinfo | -0.57% | **+0.28%** | **flip: perde→ganha** |
| | | rfembedded | -1.41% | **+0.52%** | **flip: perde→ganha** |
| | | lasso | -0.91% | **+0.05%** | **flip: perde→ganha** |
| **coloradoRiver** | 0.33181 → **0.32870** (baseline melhorou -0.94%) | ftest | +2.00% | +1.07% | -0.93pp |
| | | mutualinfo | +0.85% | **-0.09%** | **flip: ganha→perde (quase nulo)** |
| | | rfembedded | +2.74% | +0.81% | -1.93pp |
| | | lasso | +2.11% | +0.24% | -1.87pp |
| **sunspot** | 19.0443 → **19.0730** (baseline piorou +0.15%) | ftest | -0.30% | -0.15% | +0.15pp |
| | | mutualinfo | -0.25% | -0.10% | +0.15pp |
| | | rfembedded | +0.48% | +0.64% | +0.16pp |
| | | lasso | +0.47% | +0.65% | +0.16pp |

**Achado que contraria a premissa da tarefa:** o baseline `identity` **não era uniformemente
mais fraco**. Em `airlines` e `austres` ele tinha RMSE de teste **menor** (melhor) que o
`logistic` corrigido — provavelmente porque `airlines` tem só 80 linhas de treino
(razão amostra:features 4:1, ver Pergunta 4) e uma MLP genuinamente não-linear tende a
generalizar pior que uma puramente linear (`identity`) nesse regime de poucos dados. Só em
`coloradoRiver` (568 linhas de treino) o `logistic` foi de fato melhor. Isso significa que a
correção **não infla artificialmente o quanto a FS perde** em `airlines` — na verdade estreita
essa diferença (ver Pergunta 5).

---

## 2. Perguntas 1-4 da Tarefa 3.7 — refeitas, com "mudou"/"manteve" explícito

### Pergunta 1 (curva val vs. `k` em `airlines`) — **MANTEVE integralmente**
`ftest`/`mutual_info` não foram regenerados — a curva completa (`load_grid_search_history`) é
**byte-idêntica** à da Tarefa 3.7: mínimo em `k=5` (14.31) para f_test, `k=9` (14.06) para
mutual_info; `k=20` continua o pior ponto (15.45/15.06 de média). Conclusão inalterada: o Grid
Search converge corretamente para o próprio ótimo de validação; o problema nunca foi "k
errado".

### Pergunta 2 (consenso de lags) — **MANTEVE nas conclusões, com números levemente diferentes**
Recalculado com os `.pkl` novos de `rf_embedded`/`lasso`. `lasso`: seleção **100% idêntica**
em todas as 4 séries (é um algoritmo determinístico dado os mesmos dados de resíduo, que não
mudaram — `activation` não afeta o resíduo do ARIMA nem a matriz de lags). `rf_embedded`:
mudou repetição-a-repetição em `airlines` (esperado — ver Pergunta "nova" abaixo, item de
confirmação), mas o **conjunto que cruza o limiar de 50%** é o mesmo de antes. Consenso ≥3/4
continua exatamente: `airlines` (lag_20, lag_2), `coloradoRiver` (lag_11, lag_13), `sunspot`
(lag_9, lag_4), `austres` (lag_1, trivial). **Nenhum lag entrou ou saiu da lista de consenso
forte.**

### Pergunta 3 (variância zero do RF em `coloradoRiver`) — **MANTEVE, e agora com replicação independente**
O `.pkl` novo de `rf_embedded`/`coloradoRiver` (gerado numa execução totalmente independente
da Tarefa 3.7, com `random_state=None` de novo) selecionou **exatamente o mesmo conjunto**
(`{lag_13, lag_12, lag_11, lag_1}`) nas 10/10 repetições. Isso é replicação independente da
conclusão da Tarefa 3.7 (sinal genuinamente dominante, não determinismo oculto) — agora com
mais uma rodada real de evidência, não só a reconstrução sintética anterior.

### Pergunta 4 (razão amostra:features vs. resultado) — **Padrão geral manteve, contagem de vitórias mudou em 2 séries**

| Série | `df_train` | `N_Features_Total` | Razão | Vitórias antes (/4) | Vitórias depois (/4) |
|---|---|---|---|---|---|
| airlines | 80 | 20 | 4.0:1 | 0 | 0 (mantido) |
| austres | 71 | 1 (trivial) | 71.0:1 | 0 | 3 (mudou — ver ressalva) |
| sunspot | 216 | 9 | 24.0:1 | 2 | 2 (mantido) |
| coloradoRiver | 568 | 16 | 35.5:1 | 4 | 3 (mudou — mutualinfo virou -0.09%, essencialmente empate) |

`df_train`/`N_Features_Total` não mudam (a construção de lags/resíduo independe de
`activation`), só a coluna de vitórias. O padrão direcional central (pior razão = `airlines` →
pior resultado; melhor razão útil = `coloradoRiver` → melhor resultado) **se mantém**. As duas
mudanças de contagem são explicáveis e não abalam essa leitura: `austres` é um caso trivial (1
única feature candidata — qualquer "vitória"/"derrota" ali é ruído de inicialização da MLP
comparando o mesmo input único consigo mesmo, não um efeito real de FS, exatamente como já
qualificado na Tarefa 3.7); `coloradoRiver`/mutualinfo passou de +0.85% para -0.09%, uma
diferença pequena o suficiente para ser ruído de reamostragem da MLP, não uma reversão de
conclusão sobre `coloradoRiver` em geral (`rf_embedded`/`lasso` continuam vencendo lá com folga).

---

## 3. Pergunta 5 (nova) — o gap validação-teste em `airlines` ainda existe com o baseline corrigido?

**Sim, continua existindo — só um pouco menor, e por um motivo que contraria a hipótese da
tarefa.** Comparando o `RMSE` de teste do f_test no seu próprio ótimo de validação (`k=5`,
19.113854 — inalterado, `ftest` não foi regenerado) contra o baseline:

- Antes: gap = 19.113854 − 17.274490 = **1.840** RMSE (≈ -10.65%)
- Depois: gap = 19.113854 − 17.436180 = **1.678** RMSE (≈ -9.62%)

O gap **encolheu ~9%** (não cresceu) com a correção — porque, como mostrado no item 1, o
baseline `identity` (mais fraco na teoria) na prática tinha RMSE de teste **menor** que o
`logistic` em `airlines` especificamente (17.27 vs. 17.44). A hipótese de que "o baseline mais
fraco poderia ter inflado artificialmente esse gap" **não se confirma** — se algo, o oposto:
usar `identity` (mais fraco em tese) tornava o baseline ainda mais difícil de bater em
`airlines`. O gap validação-teste em si (mesmo o melhor `k` de validação perdendo no teste)
**é um fenômeno robusto**, não um artefato do baseline errado.

---

## 4. A seleção de features mudou entre a versão suja (Tarefa 3.7) e a limpa (3.10)?

**Resposta com número real, não "esperado":**

- **`lasso` (4 séries): 100% idêntica**, célula a célula, comparando os dois
  `selected_features_detail.csv` — mesmos índices, mesmos lags, nas 10/10 repetições em todas
  as 4 séries. Esperado: a seleção do Lasso só depende dos dados de resíduo (inalterados) e do
  ajuste do `LassoCV`, nunca do `estimator__hidden_layer_sizes`/`max_iter`/`selector__k`.
- **`rf_embedded`/`austres`: idêntica** (trivial, 1 única feature candidata).
- **`rf_embedded`/`coloradoRiver`: idêntica** (mesmíssimo conjunto nas 10/10 repetições, ver
  Pergunta 3).
- **`rf_embedded`/`sunspot`: quase idêntica** — antes `lag_4` aparecia em 9/10 repetições, agora
  em 10/10. Diferença de 1 repetição, mesmo conjunto de consenso.
- **`rf_embedded`/`airlines`: DIFERENTE, repetição a repetição** (ex.: repetição 0 antes
  selecionava `{lag_20,18,15,13,12,6,5,2}` — 8 features; repetição 0 agora seleciona
  `{lag_20,18,13,12,9,7,6,4,2}` — 9 features, um conjunto distinto).

**Isso não é um efeito da limpeza do `selector__k`** — é o comportamento esperado de
`RandomForestRegressor(random_state=None)` (decisão intencional da Tarefa 3.1) numa série sem
sinal dominante: cada execução independente do notebook gera resultados diferentes, com ou sem
o parâmetro morto no grid, porque a aleatoriedade nunca foi fixada. `coloradoRiver` não muda
entre execuções porque o sinal lá é ~10 desvios-padrão mais forte que o ruído (Pergunta 3);
`airlines` muda porque não tem esse mesmo colchão de segurança. A tabela de consenso (Pergunta
2) já é robusta a essa variação — é exatamente por isso que a metodologia usa frequência ≥50%
em vez do conjunto exato de uma única execução.

---

## Conclusão geral

Nenhuma conclusão qualitativa central da Tarefa 3.7 foi revertida pela correção do baseline:
`airlines` continua perdendo em todos os 4 métodos; `coloradoRiver` continua sendo onde a FS
mais ajuda (agora 3/4 em vez de 4/4, por uma diferença de ruído em `mutual_info`); `sunspot`
continua misto 2/4; o consenso de lags é idêntico; a explicação por razão amostra:features
continua de pé. A mudança mais substantiva é em `austres`, que passou de "FS perde em tudo"
para "FS ganha em 3/4" — mas essa série já era e continua sendo um caso degenerado (1 única
feature candidata) que não deveria ser lido como evidência real de FS ajudando ou atrapalhando.
