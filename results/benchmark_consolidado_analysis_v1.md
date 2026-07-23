# Benchmark Consolidado — 5 Famílias × 5 Métodos de FS (Tarefa 8)

Análise final cruzando toda a matriz de benchmark: `ARIMA` (referência linear pura, sem FS),
`ARIMA-MLP`, `MLP`, `SVR`, `ARIMA-SVR` (cada uma com `f_test`, `mutual_info`, `rf_embedded`,
`lasso`, `rfecv`), 4 séries (`airlines`, `austres`, `coloradoRiver`, `sunspot`). Nenhum `.pkl`
novo gerado — 100% leitura/agregação dos CSVs já produzidos pelos portões de validação
anteriores (Tarefas 5-gate a 7.2).

Tabela mestre completa: `results/benchmark_consolidado_v1.csv` (84 linhas: 4 séries × [1 ARIMA
+ 5 métodos × 4 famílias com FS], `austres` marcada `Trivial=True` mas mantida nos dados).

---

## Resumo executivo (4 achados centrais)

1. **`airlines` não é um caso universal de "FS atrapalha" — é específico das famílias
   híbridas e do `SVR` single.** Das 4 famílias com FS aplicável, **só `MLP` single tem ganho
   líquido médio positivo** em `airlines` (+15,36%); as outras 3 perdem em média
   (`ARIMA-MLP` -10,01%, `ARIMA-SVR` -12,23%, `SVR` -13,63%). O melhor resultado individual do
   benchmark inteiro em `airlines` é `MLP × f_test`, +21,97%.
2. **`coloradoRiver` NÃO ganha em todas as famílias — `SVR` single é a exceção real.**
   `ARIMA-MLP` (4/5 métodos vencem), `MLP` (4/5) e `ARIMA-SVR` (5/5, único caso de vitória
   unânime do benchmark) mostram ganho consistente; **`SVR` single vence em só 1/5 métodos**
   (`lasso`, +0,31%) e tem ganho líquido médio **negativo** (-13,57%). A hipótese original
   ("ganha nas famílias híbridas/SVR") só é parcialmente verdadeira — o padrão é híbridas
   (ambas) + `MLP` single, não `SVR` single.
3. **Nenhum método vence "em geral" de forma limpa — depende de como se mede "vencer".**
   Por frequência (quantas vezes `PctGain>0`, de 12 combinações família×série não-triviais):
   `rfecv` vence mais (7/12), seguido de `lasso` (6/12). Por magnitude média: só `f_test` tem
   média positiva (+0,93%); todos os outros são negativos em média (`rfecv` -0,52%,
   `mutual_info` -2,85%, `lasso` -7,43%, `rf_embedded` -8,50%, o pior). `rf_embedded` tem a
   maior variância de resultado (de +40,36% em `MLP`/`coloradoRiver` a -60,88% em
   `MLP`/`sunspot`) — o método mais dependente de contexto do benchmark inteiro.
4. **Consenso de lags cruza famílias de verdade em `lag_12`/`lag_13` (`airlines`) e
   `lag_11`/`lag_12` (`coloradoRiver`)** — presentes como "consenso forte" (≥3 dos 5 métodos)
   nas 4 famílias simultaneamente, incluindo `lag_12`/`airlines` com votação **unânime (5/5
   métodos)** em `MLP` e `SVR` single. Isso é evidência mais robusta de relevância genuína do
   que qualquer resultado de RMSE isolado, porque é cega ao viés de otimização de cada família.

---

## Tabela mestre resumida (`PctGain` médio por Família × Série, séries não-triviais)

| Família | airlines | coloradoRiver | sunspot | Média geral |
|---|---|---|---|---|
| ARIMA-MLP | -10.01% | +0.47% | +0.06% | -3.16% |
| MLP | **+15.36%** | **+24.46%** | **+2.75%** | **+14.19%** |
| SVR | -13.63% | -13.57% | -21.66% | -16.29% |
| ARIMA-SVR | -12.23% | **+7.87%** | -0.90% | -1.75% |

(`austres` excluída desta tabela — caso trivial, 1 único lag candidato em todas as famílias,
qualquer "vitória"/"derrota" ali é ruído de inicialização, não efeito real de FS.)

---

## Respostas à Parte B

### 1. `airlines`: universal ou específico?

**Específico.** 1 de 4 famílias com ganho líquido médio positivo: `MLP` single (+15,36%). As
outras 3 (`ARIMA-MLP` -10,01%, `ARIMA-SVR` -12,23%, `SVR` -13,63%) confirmam o padrão "FS não
ajuda em `airlines`" — mas `MLP` é uma exceção real e forte, não ruído (todos os 4 métodos
não-triviais de `MLP` em `airlines` são positivos: `f_test` +21,97%, `mutual_info` +15,08%,
`rf_embedded` +18,27%, `rfecv` +17,10%; só `lasso` fica marginal, +4,38%).

### 2. `coloradoRiver`: ganha em todas as famílias?

**Não — `SVR` single é a exceção.** Vitórias por família (métodos com `PctGain>0`, de 5):
`ARIMA-MLP`=4/5, `ARIMA-SVR`=5/5 (único caso unânime do benchmark), `MLP`=4/5, **`SVR`=1/5**
(ganho líquido médio -13,57%, o único negativo em `coloradoRiver`). A leitura correta não é
"híbridas + SVR ganham" — é "3 das 4 famílias ganham de forma consistente; `SVR` single é a
única exceção real, e perde por margem considerável nos outros 4 métodos (`f_test` -11,02%,
`mutual_info` -23,38%, `rf_embedded` -15,08%, `rfecv` -18,66%)".

### 3. Ranking geral

Melhor combinação por série (excluindo `austres`):
- `airlines`: **`MLP` × `f_test`**, PctGain = +21,97%
- `coloradoRiver`: **`MLP` × `rf_embedded`**, PctGain = +40,36% (o melhor resultado absoluto
  do benchmark inteiro)
- `sunspot`: **`ARIMA-SVR` × `rfecv`**, PctGain = +1,43% (modesto — `sunspot` é a série com os
  menores ganhos possíveis em geral)

Não há um método universal. Por **frequência de vitória** (12 combinações família×série
possíveis, excluindo `austres`): `rfecv` vence 7/12, `lasso` 6/12, `rf_embedded` 5/12,
`f_test` 4/12, `mutual_info` 3/12. Por **magnitude média**: só `f_test` fica positivo
(+0,93%); `rfecv` -0,52%, `mutual_info` -2,85%, `lasso` -7,43%, `rf_embedded` -8,50%.
`rf_embedded` vence com menos frequência mas, quando ganha, ganha muito (+40,36% em
`MLP`/`coloradoRiver`) — e quando perde, perde muito (-60,88% em `MLP`/`sunspot`). O resultado
**é fortemente dependente de contexto** — nenhum método domina de forma limpa em frequência
E magnitude ao mesmo tempo.

### 4. Consenso de lags

**Cruzando os 5 métodos dentro de cada família** (consenso forte = ≥3/5 métodos concordando
na mesma repetição-moda), e depois **cruzando as 4 famílias entre si**:

- **`airlines`**: `lag_12` e `lag_13` em **consenso forte nas 4 famílias simultaneamente**.
  `lag_12` tem votação **unânime (5/5 métodos)** em `MLP` e `SVR` single — o achado mais
  robusto para esta série. `lag_20` só aparece como consenso em `ARIMA-MLP`/`ARIMA-SVR` (não
  em `MLP`/`SVR` single), e em ambas as famílias híbridas o voto de `lasso` está presente —
  como a Tarefa 3.6 já provou que `lasso`/`airlines`/`ARIMA-MLP` é fallback (não seleção
  genuína), `lag_20` deve ser lido com ressalva; `lag_12`/`lag_13` NÃO dependem do voto de
  `lasso` em `ARIMA-MLP` (confirmado lendo os métodos individualmente), então são evidência
  mais confiável.
- **`coloradoRiver`**: `lag_12` em consenso forte nas 4 famílias; `lag_11`/`lag_13` em 3 das 4
  (mesmo padrão robusto já visto na análise por família na Tarefa 3.7/8-gate).
- **`sunspot`**: `lag_9`, `lag_1`, `lag_2` em consenso forte nas 4 famílias; `lag_4` em 3 das 4.
- **`austres`**: `lag_1` em consenso "forte" nas 4 famílias — mas é o único lag candidato
  (trivial, sem valor evidencial).

### 5. Efeito do `gamma='auto'` — `SVR` single vs. `ARIMA-SVR`

| Método | `SVR`/airlines PctGain (NFeatures) | `ARIMA-SVR`/airlines PctGain (NFeatures) |
|---|---|---|
| `rf_embedded` | -13,87% (1) | -32,13% (9) |
| `lasso` | -15,65% (7) | -27,16% (1) |
| `rfecv` | -38,64% (2) | -1,84% (18) |

**Evidência mista, não uma confirmação limpa da hipótese.** Para `rfecv`, o padrão é
consistente com a hipótese: `SVR` reduz agressivamente (2 features, `gamma` efetivo grande) e
perde muito (-38,64%); `ARIMA-SVR` quase não reduz (18 features) e perde pouco (-1,84%). Mas
`rf_embedded` e `lasso` vão na direção **oposta** entre si: `SVR`/`rf_embedded` mantém só 1
feature mas perde MENOS (-13,87%) que `ARIMA-SVR`/`rf_embedded`, que mantém 9 features e perde
MAIS (-32,13%). Não dá para isolar o efeito de `gamma` da magnitude da redução de forma limpa
com os dados disponíveis (confirma a necessidade da nota provisória continuar em aberto — não
é um efeito comprovadamente dominante em todos os métodos, só plausível para alguns).

### 6. ARIMA puro no ranking geral

Comparado por RMSE absoluto contra as 20 combinações família×método (excluindo `austres`),
`ARIMA` fica **consistentemente no meio da tabela, nunca no topo nem no fundo**:

- `airlines`: rank **11 de 21** — perde para toda a família `ARIMA-SVR` e para `ARIMA-MLP`/`rfecv`/`ftest`, mas vence a maioria de `MLP`/`SVR`.
- `coloradoRiver`: rank **15 de 21** — perde para TODA a família `SVR` e TODA a `MLP` (10 combinações consecutivas melhores), só vence parte de `ARIMA-MLP`/`ARIMA-SVR`.
- `sunspot`: rank **11 de 21** — perde para toda a `MLP` e para vários `ARIMA-MLP`/`ARIMA-SVR`, vence toda a `SVR`.

`ARIMA` funciona bem como **piso de comparação para `SVR`/`MLP` single em `coloradoRiver`**
(ambas as famílias o superam completamente ali) e como **teto informal para `SVR`/`ARIMA-SVR`
em `sunspot`** (nenhuma variante de `SVR` supera o ARIMA puro nessa série) — mas não é nem piso
nem teto absoluto do benchmark; a posição muda por série.

---

## Limitações metodológicas conhecidas

1. **Gap validação-teste em `airlines`/híbrido-MLP** (Tarefa 3.7/3.10): mesmo o `k` ótimo de
   validação em `f_test`/`mutual_info` no híbrido `ARIMA-MLP` perde para o baseline no teste —
   não é um problema de escolha de hiperparâmetro, é generalização.
2. **`gamma='auto'` não isolado** (PLANO_ARQUITETURA.md Seção 1.11, nota **provisória**, não
   definitiva): `gamma=1/n_features` muda junto com a seleção de features nas famílias `SVR`/
   `ARIMA-SVR`, confundindo o efeito de FS com o de largura de kernel — Pergunta 5 acima mostra
   que o efeito não é limpo nem sempre na direção esperada. Se a decisão do orientador mudar,
   `SVR` e `ARIMA-SVR` precisam ser re-rodadas.
3. **`austres` é caso trivial em todas as 5 famílias** (`N_Features_Total=1`, `lag_size='auto'`
   via PACF) — qualquer "vitória"/"derrota" ali é ruído de inicialização comparando o mesmo
   único input consigo mesmo, nunca um efeito real de FS. Mantida na tabela mestre bruta, mas
   excluída de toda agregação/ranking desta análise.
4. **`lasso` frequentemente cai no fallback de zero-features** (Tarefa 3.1/3.6): quando o
   `SelectFromModel` zera todos os coeficientes, o fallback mantém a feature de maior
   `|coeficiente|` — isso já foi confirmado como o caso em `ARIMA-MLP`/`lasso` para `airlines`,
   `austres`, `coloradoRiver` (não `sunspot`). O mesmo mecanismo provavelmente afeta `lasso` nas
   outras 3 famílias sempre que `NFeatures=1`, mas isso **não foi reconfirmado individualmente
   para `MLP`/`SVR`/`ARIMA-SVR` nesta tarefa** (só a inferência estrutural via `NFeatures=1`) —
   fica como pergunta em aberto.
5. **`model_exec` difere por família por desenho, não por inconsistência**: `MLP`/`ARIMA-MLP`
   usam 10 repetições (estocástico); `SVR`/`ARIMA-SVR` usam 1 (determinístico) — CLAUDE.md
   Seção 3.4. Isso significa que os números de `SVR`/`ARIMA-SVR` têm zero variância amostral
   reportada (só 1 execução), enquanto `MLP`/`ARIMA-MLP` são médias de 10 — as duas famílias
   não são diretamente comparáveis em termos de robustez estatística do número reportado.
6. **Nenhum teste de hipótese formal foi aplicado** (CLAUDE.md Seção 3.4) — todos os "achados"
   acima são padrões descritivos sobre 4 séries e no máximo 5 repetições por combinação, não
   conclusões com significância estatística testada.

---

## Perguntas em aberto reveladas por esta análise

1. Vale a pena reconfirmar o fallback do `lasso` para as famílias `MLP`/`SVR`/`ARIMA-SVR`
   individualmente (mesma técnica de reconstrução da Tarefa 3.6), já que a inferência aqui foi
   só estrutural (`NFeatures=1`)?
2. `lag_12`/`airlines` e `lag_11`-`lag_12`/`coloradoRiver` — como achados de consenso robustos
   entre 4 famílias e 5 métodos, esses lags mereceriam uma seção dedicada e talvez uma
   interpretação de domínio (o que esses lags representam fisicamente/sazonalmente nas séries)
   para a dissertação, algo que não foi feito aqui (fora do escopo de uma análise puramente
   numérica).
3. `sunspot` tem os menores ganhos possíveis em qualquer família/método (o melhor resultado da
   série inteira é só +1,43%) — vale investigar se isso é uma característica genuína da série
   (resíduo já bem ajustado pelo ARIMA) ou um artefato de configuração ainda não identificado.
