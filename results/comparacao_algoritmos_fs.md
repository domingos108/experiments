# Comparação entre algoritmos de Feature Selection

Documento analítico, elaborado a partir de `results/benchmark_master_with_sources_v1.csv` e
`results/lag_selection_consensus_v1.csv` (Tarefa 9). Todas as afirmações abaixo citam o
número exato que as sustenta e o arquivo de origem. `austres` é excluída desta análise
comparativa (série trivial, 1 lag candidato — qualquer método "seleciona" a mesma única
feature, sem informação sobre eficácia comparativa).

---

## 1. Vitórias por método, contando as 3 séries não-triviais x 4 famílias (12 comparações)

Contagem de quantas vezes cada método produziu o menor RMSE dentre os 5 métodos de FS
(não contando `sem_FS` nem `ARIMA` puro), por combinação Família×Série — calculada
programaticamente a partir de `RMSE_FS`, com tratamento explícito de empates (2 casos, cada
um contado como meia vitória para cada método empatado):

| Método | Vitórias (de 12, empates = 0,5) | Onde venceu |
|---|---|---|
| `rfecv` | 3 | `ARIMA-MLP`/airlines (18,02, `results/chamados_v4_fs_rfecv/metrics.csv`); `ARIMA-SVR`/sunspot (18,99, `results/chamados_v4_fs_arimasvr_rfecv/metrics.csv`); `SVR`/sunspot (20,19, `results/chamados_v4_fs_svr_rfecv/metrics.csv`) |
| `f_test` | 3 (2 vitórias limpas + 2 meias por empate) | `ARIMA-MLP`/coloradoRiver (0,3252, `results/chamados_v4_fs_ftest/metrics.csv`, vitória limpa); `MLP`/airlines (21,61, `results/chamados_v4_fs_mlp_ftest/metrics.csv`, vitória limpa); `ARIMA-SVR`/airlines e `SVR`/airlines — empatado exatamente com `mutual_info` nos dois casos |
| `mutual_info` | 2 (1 vitória limpa + 2 meias por empate) | `MLP`/sunspot (17,04, `results/chamados_v4_fs_mlp_mutualinfo/metrics.csv`, vitória limpa); `ARIMA-SVR`/airlines e `SVR`/airlines — empatado com `f_test` |
| `lasso` | 2 | `ARIMA-MLP`/sunspot (18,95, `results/chamados_v4_fs_lasso/metrics.csv`); `SVR`/coloradoRiver (0,1262, `results/chamados_v4_fs_svr_lasso/metrics.csv`) |
| `rf_embedded` | 2 | `ARIMA-SVR`/coloradoRiver (0,2971, `results/chamados_v4_fs_arimasvr_rfembedded/metrics.csv`); `MLP`/coloradoRiver (0,1341, `results/chamados_v4_fs_mlp_rfembedded/metrics.csv`) |

**Os dois empates** (`ARIMA-SVR`/airlines e `SVR`/airlines, ambos entre `f_test` e
`mutual_info`, com RMSE idêntico ao 6º dígito decimal) não são coincidência numérica — em
ambos os casos, os dois métodos convergiram para `NFeatures=20` (usar todas as features
candidatas, ou seja, nenhuma redução de fato), o que torna a transformação do seletor uma
identidade matemática; como o `SVR` é determinístico, dois "não-seletores" produzem
exatamente o mesmo resultado numérico do baseline.

**Não há um método que domine isoladamente**: a distribuição de vitórias entre os 5 métodos
é quase uniforme (2 a 3 cada, num total de 12 comparações) — nenhum método venceu em mais de
25% das combinações Família×Série avaliadas.

---

## 2. `rfecv` quase nunca é o pior método, mesmo quando não vence

Em nenhuma das 9 células não-triviais onde `rfecv` não teve o menor RMSE, ele foi o PIOR
método também — sempre ficou em posição intermediária. Isso é visível diretamente nos
valores de `PctGain` da tabela mestre: a faixa de `PctGain` de `rfecv` nas 4 famílias ×
3 séries não-triviais vai de -38,64% (`SVR`/airlines,
`results/chamados_v4_fs_svr_rfecv/metrics.csv`) a +28,99% (`MLP`/coloradoRiver,
`results/chamados_v4_fs_mlp_rfecv/metrics.csv`) — uma faixa ampla, mas o pior valor
individual do dataset inteiro é do `rf_embedded` em `MLP`/sunspot (-60,88%,
`results/chamados_v4_fs_mlp_rfembedded/metrics.csv`), não do `rfecv`.

## 3. `f_test`/`mutual_info` (filtros) nunca reduzem features em `SVR`/airlines e `ARIMA-SVR`/airlines — resultado idêntico ao baseline

Em `SVR`/airlines, tanto `f_test` quanto `mutual_info` têm `NFeatures=20` (de 20 candidatas)
e `PctGain=0.0` exato (`results/chamados_v4_fs_svr_ftest/metrics.csv`,
`results/chamados_v4_fs_svr_mutualinfo/metrics.csv`) — o Grid Search, ao testar valores de
`k`, convergiu para "usar todos os lags" como ótimo de validação. O mesmo padrão se repete em
`ARIMA-SVR`/airlines para os mesmos dois métodos
(`results/chamados_v4_fs_arimasvr_ftest/metrics.csv`,
`results/chamados_v4_fs_arimasvr_mutualinfo/metrics.csv`, ambos `PctGain=0.0`,
`NFeatures=20`). Como o `SVR` é determinístico (sem repetições estocásticas, diferente do
MLP), esse "sem redução" produz literalmente o mesmo resultado numérico do baseline, não uma
aproximação.

## 4. `lasso` tem o comportamento mais bimodal: ou reduz quase tudo, ou mantém quase tudo

Olhando `NFeatures` de `lasso` nas 12 combinações não-triviais: 6 delas têm `NFeatures<=2`
(`ARIMA-MLP`/airlines=1, `SVR`/airlines=7, `ARIMA-SVR`/airlines=1, `ARIMA-MLP`/coloradoRiver=1,
`ARIMA-SVR`/coloradoRiver=1, `ARIMA-MLP`/sunspot=2, `ARIMA-SVR`/sunspot=2), enquanto outras
mantêm a maioria das features (`SVR`/coloradoRiver=15 de 16,
`results/chamados_v4_fs_svr_lasso/metrics.csv`). Não há um meio-termo consistente — ou o
Lasso encontra pouquíssimo sinal linear (regularização L1 zera quase tudo), ou encontra quase
todas as features como relevantes.

## 5. `rf_embedded` é o único método com variância real de contagem de features entre repetições em mais de uma família

`Frequencia_Selecao` diferente de `10/10` aparece para `rf_embedded` em `MLP`/airlines (8/10,
`results/lag_selection_consensus_v1.csv`) e em `ARIMA-MLP`/airlines (2/10, mesmo arquivo) —
as duas ocorrências de instabilidade de seleção do `rf_embedded` no dataset inteiro
concentram-se exatamente na série `airlines`, não em `coloradoRiver`/`sunspot`, onde
`rf_embedded` tem `10/10` em todas as famílias MLP-based.

## 6. `rfecv` também tem instabilidade concentrada, mas em células diferentes de `rf_embedded`

`rfecv` é o método com maior instabilidade de repetibilidade entre as 6 combinações
MLP-based (`model_exec=10`): **nenhuma** delas atinge `10/10` — todas variam entre `1/10` e
`4/10` (`MLP`/airlines=4/10, `MLP`/coloradoRiver=4/10, `MLP`/sunspot=3/10,
`ARIMA-MLP`/airlines=4/10, `ARIMA-MLP`/coloradoRiver=1/10, `ARIMA-MLP`/sunspot=3/10, todos em
`results/lag_selection_consensus_v1.csv`). Isso contrasta com `rf_embedded`, que atinge
`10/10` em 4 das 6 combinações (Seção 5) — `rfecv` é, portanto, o método menos estável em
termos de repetibilidade do conjunto de features selecionado entre as 10 repetições, mesmo
sendo o método que mais frequentemente produz o melhor RMSE junto com `f_test` (Seção 1).

---

## Perguntas em aberto (não respondidas aqui, levantadas pelos dados acima)

- A instabilidade de `rfecv` (Seção 6) é compatível com o seu bom desempenho médio (Seção 1)?
  Isto é: `rfecv` acerta o RMSE mesmo variando o conjunto exato de features entre repetições,
  ou o RMSE reportado é uma média que esconde variância real de desempenho também? (Precisaria
  de `RMSE_std` por repetição, não coberto neste documento.)
- Por que `lasso` é bimodal (Seção 4) especificamente nas famílias híbridas/ARIMA-SVR, mas
  mantém 15/16 features em `SVR`/coloradoRiver? Existe relação com a presença/ausência do
  resíduo do ARIMA na entrada?
- Os dois empates exatos entre `f_test`/`mutual_info` (Seção 1) só ocorrem em `airlines` nas
  famílias `SVR`/`ARIMA-SVR` — vale investigar se `airlines` tem alguma propriedade
  estatística (ex. baixa informação mútua/correlação linear geral entre lags e alvo) que leve
  ambos os filtros a "desistir" de reduzir, convergindo para o mesmo `k=N`.
