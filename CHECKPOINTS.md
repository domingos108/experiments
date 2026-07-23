# CHECKPOINTS.md — Estado do Projeto (Feature Selection Nativo)

Este documento existe para permitir retomar o trabalho após uma pausa ou perda de sessão de
chat, sem depender do histórico de conversa. Deve ser mantido atualizado a cada tarefa
concluída (ou pausada) — é o complemento "estado atual" ao lado de CLAUDE.md (regras) e
PLANO_ARQUITETURA.md (arquitetura/roadmap).

**Última atualização:** 2026-07-22, ao final da Tarefa 7.2 — auditoria de integridade automatizada da matriz 5×5 completa (486 PASS, 0 FAIL, 0 ATENÇÃO).
**Branch/estado do Git no momento desta pausa:** `joao_lucas_experiments`. Working tree com
mudanças das Tarefas 3.1 a 3.4 **não commitadas** (ver Seções 2, 2b e 2c) — o commit `98eccef`
("Runbook and checkpoints added") já continha o estado da Tarefa 3 completo; tudo abaixo dele
é novo.

---

## 1. Tabela de Checkpoints

| # | Tarefa | Status | Entregáveis principais |
|---|---|---|---|
| 1 | Scaffold `TimeSeriesFeatureSelector` (f_test, mutual_info) + fix `fs_lag_size` | ✅ Concluída | `src/model/feature_selection.py`, `resolve_lag_size()` em `grid_search_exp.py`, `config.py` |
| 2 | Integração via Pipeline em `arima_mlp_ftest.ipynb` | ✅ Concluída | Primeira execução real: `sunspot`+`airlines`, `f_test`, `experiment_id=chamados_v2_fs_ftest` |
| 2.5 | Triagem de notebooks suspeitos + fix `metrics.py` (bug Windows) | ✅ Concluída | `.gitignore` para `exploratory_analysis/`; `metrics.py` corrigido e testado |
| 2.6 | Reorganização de notebooks + runbook | ✅ Concluída (com 1 lacuna, ver Seção 3) | `arima_mlp.ipynb` restaurado como baseline; `RUNBOOK.md` criado; nota de `diff_kpss` no CLAUDE.md |
| 3 | `rf_embedded`/`lasso` (top-k uniforme) + 4 notebooks dedicados | ✅ Concluída, **nada executado** | `experiment_id` propostos `chamados_v3_fs_*` — **superados pela Tarefa 3.1**, nunca executados |
| 3.1 | Reversão p/ `SelectFromModel` nativo + registro de features + comparação lado a lado | ✅ Concluída, **nada executado ainda** | Ver Seção 2 abaixo — código/testes/docs prontos, execução real fica para a próxima sessão |
| 3.2 | Fluxo notebook-only (Run All, sem terminal) + investigação PACF `austres.txt` | ✅ Concluída, **nada executado ainda** | Ver Seção 2b abaixo — 4 notebooks reestruturados (7 células), `notebook/compare_fs_results.ipynb` novo, `src/utils/copy_pretrained_linear_model.py` novo |
| 3.3 | Diagnóstico: `compare_fs_results.ipynb` corrompido? Grid de `k` ampliado foi avaliado de verdade? | ✅ Concluída (só investigação, sem correção) | **Notebook NÃO estava corrompido** (JSON bruto validado + rodado via papermill, sem erro) — corrupção relatada não reproduzida. **Histórico completo do grid confirmado como não-persistido** (`_search_params()` descarta tudo exceto o vencedor) — motivou a Tarefa 3.4 |
| 3.4 | Persistir histórico completo do Grid Search (combinação × erro de validação) | ✅ Concluída, **nada executado em experimento real ainda** | Ver Seção 2c abaixo — `grid_search_history` no `.pkl`, `load_grid_search_history()`, 1 bug real corrigido via code-review |
| 3.5 | Fix `pytest` não coletava 4 arquivos rodando de `experiments/` | ✅ Concluída | `pythonpath = .` em `pytest.ini` — causa raiz: `experiments/` não entrava em `sys.path` de forma determinística sem essa opção |
| 3.6 | Instrumentar fallback zero-features (`rf_embedded`/`lasso`) + investigar `chamados_v4_fs_lasso` real | ✅ Concluída | `fallback_triggered_` em `TimeSeriesFeatureSelector`; reconstrução retroativa confirmou fallback em `airlines`/`austres`/`coloradoRiver` (não em `sunspot`) |
| 3.7 | Análise aprofundada dos resultados de `chamados_v4_fs_*` (5 perguntas) | ✅ Concluída (só leitura/análise) | `results/chamados_v4_fs_analysis_report.md` novo — curva val-vs-k, consenso de lags, variância RF em `coloradoRiver`, razão amostra:features |
| 3.8 | Fechar 2 perguntas em aberto da 3.7: grid MLP baseline-vs-FS, `selector__k` morto | ✅ Concluída (só investigação) | Achado real: baseline `.pkl` desatualizado (`activation='identity'` vs. `'logistic'` no código); `selector__k` morto confirmado em `rf_embedded`+`lasso` |
| 3.9 | Corrigir baseline desatualizado (`1amv1`) + limpar `selector__k` morto | ✅ Preparado e **executado pelo pesquisador** | `.pkl` `identity` arquivado; `arima_mlp.ipynb`/`arima_mlp_rf_embedded.ipynb`/`arima_mlp_lasso.ipynb` rodados manualmente — `1amv1`/`rfembedded`/`lasso` regenerados |
| 3.10 | Reanálise completa com baseline `1amv1` corrigido (mesmas 5 perguntas da 3.7 + 1 nova) | ✅ Concluída (só leitura/análise) | `results/chamados_v4_fs_analysis_report_v2.md` novo — nenhuma conclusão qualitativa central revertida; `austres` mudou de "perde em tudo" para "ganha em 3/4" (caso trivial, ruído) |
| 4 | `rfecv` (último método obrigatório do arsenal) | ✅ Concluída, **notebook pronto, não executado** | `strategy='rfecv'` em `feature_selection.py` (TDD completo), `arima_mlp_rfecv.ipynb`, `chamados_v4_fs_rfecv` proposto — 8 agentes code-review, 3 correções aplicadas |
| 5 | Generalizar FS para `MLP` single (`SKlearnModel`), 1ª família fora de `Additive` | ✅ Concluída, **5 notebooks prontos, nenhum executado** | Compatibilidade estrutural confirmada com teste real (`tests/model/test_single_ml_model_exp.py`); `lag_size='auto'` medido idêntico ao híbrido; 5 notebooks em `notebook/single_models/mlp_<estrategia>.ipynb`, `chamados_v4_fs_mlp_<estrategia>` |
| 5.1 | Corrigir baseline `1mlp` desatualizado (`diff_kpss`), mesmo padrão da 3.9 | ✅ Preparado e **executado pelo pesquisador** | `.pkl` `diff_kpss=True` arquivado; `mlp_exec.ipynb` rodado manualmente — `1mlp` regenerado com `diff_kpss=False` (confirmado, Tarefa 5.2) |
| 5.2 | Investigar "ATENÇÃO" falso na verificação pós-execução de `mlp_exec.ipynb` | ✅ Concluída (causa raiz certa, não hipótese) | `airlines_1amv1.pkl` nunca foi tocado — falso positivo por BOM (UTF-8) no snapshot de hash não removido por `read_text(encoding='utf-8')`; reproduzido e confirmado com `encoding='utf-8-sig'` |
| 6-gate | Portão de validação MLP single (estrutura + comparação rápida) antes da Tarefa 6 (SVR) | ✅ Concluída | 20 `.pkl` validados (`activation`/`diff_kpss` corretos, sem contaminação `amv1*`); guarda 1-feature do `rfecv` confirmada em produção (`austres`); `compare_fs_vs_baseline.py` generalizado (`baseline_model_name`/`linear_model_name_to_exclude`, retrocompatível); `results/chamados_v4_fs_mlp_comparison.csv` + `notebook/compare_fs_results_mlp.ipynb` novos |
| 6 | Generalizar FS para `SVR` single (`SKlearnModel`), 2ª família fora de `Additive` | ✅ Concluída e **executada pelo pesquisador** | Compatibilidade confirmada com teste real (`TestSKlearnModelAcceptsPipelineWithSVR`); `model_exec=1`/`diff_kpss=True` (diferente do MLP, igual ao baseline `svr_exec.ipynb`); `lag_size='auto'` medido idêntico; grid `C`/`gamma`/`kernel`/`epsilon`/`tol` extraído do baseline; 5 notebooks em `notebook/single_models/svr_<estrategia>.ipynb`, `chamados_v4_fs_svr_<estrategia>` |
| 7-gate | Portão de validação SVR single (estrutura + comparação rápida) antes da Tarefa 7 (ARIMA-SVR) | ✅ Concluída | `svr_exec.ipynb` confirmado no-op seguro (mtime idêntico ao pré-sessão); 20 `.pkl` validados (grid `C`/`gamma`/`kernel`/`epsilon`/`tol` com paridade ao baseline, `diff_kpss=True`, `n_reps=1`, sem contaminação de outras famílias); 5 baselines protegidos intactos; `results/chamados_v4_fs_svr_comparison.csv` + `notebook/compare_fs_results_svr.ipynb` novos |
| 7 | Generalizar FS para `ARIMA-SVR`, última família da matriz (5×5) | ✅ Concluída e **executada pelo pesquisador** | Compatibilidade `Additive`+`Pipeline([selector,SVR])` confirmada com teste real (`tests/model/test_hybrid_system_exp.py`); `lag_size='auto'` medido idêntico ao ARIMA-MLP (resíduo independe do estimador); grid extraído de `arima_svr.ipynb`; 5 notebooks em `notebook/residual_hydridsystem/arima_svr_<estrategia>.ipynb`, `chamados_v4_fs_arimasvr_<estrategia>`; nota PROVISÓRIA sobre `gamma='auto'` documentada (PLANO_ARQUITETURA.md Seção 1.11) |
| 8-gate | Portão de validação final ARIMA-SVR — **matriz 5×5 completa** (5 famílias × 5 métodos) | ✅ Concluída | 20 `.pkl` validados (grid `C`/`gamma`/`kernel`/`epsilon`/`tol` com paridade total, `diff_kpss=False`, `linear_model_name='1arima'`, `n_reps=1`, sem contaminação); 5 baselines protegidos intactos; `results/chamados_v4_fs_arimasvr_comparison.csv` + `notebook/compare_fs_results_arimasvr.ipynb` novos; `tests/model/test_hybrid_system_exp.py` confirmado como arquivo novo (2 testes, pré-check da Tarefa 7, sem mudança de produção) |
| 7.2 | Auditoria de integridade metodológica única, reutilizável e permanente (5 baselines + 20 experimentos) | ✅ Concluída | `src/utils/audit_experiment_integrity.py` novo (TDD, 33 testes); `results/integrity_audit_report_v1.md` — **486 PASS, 0 FAIL, 0 ATENÇÃO, 8 N/A** (N/A = baseline ARIMA, sem `experiment_params`/estimador sklearn); `data/result/chamados_baseline_reference_hashes.json` novo (referência de hash reutilizável); 1 bug real encontrado e corrigido no próprio script de auditoria durante a execução (spy de `LassoCV` como subclasse quebrava `clone()` interno do `SelectFromModel`, mascarado por um `except Exception` que engolia o erro — corrigido para spy funcional, igual ao já usado para `RFECV`) |

---

## 2. O que a Tarefa 3.1 entregou (não commitado ainda)

**Parte A — Reversão `rf_embedded`/`lasso` para `SelectFromModel` nativo:**
`src/model/feature_selection.py` — `rf_embedded`/`lasso` usam `SelectFromModel(estimator,
threshold=None)`; nº de features emerge do ajuste (não mais `k` fixo). Threshold `None`
resolve para `'mean'` das importâncias (RF) e `~1e-5` (Lasso, detecção de L1 pelo sklearn).
`f_test`/`mutual_info` inalterados (`k` continua pesquisável). **Bug crítico encontrado e
corrigido durante o code-review**: `SelectFromModel(LassoCV(...))` podia selecionar **zero**
features quando o alvo não tem nenhuma relação linear com X (residuo ARIMA quase-ruído-branco),
quebrando `MLPRegressor.fit()` a jusante — corrigido com fallback determinístico (mantém a
feature de maior `|coeficiente|`/importância quando o corte automático zera tudo).

**Parte B — `fs_lag_size` removida (não setada como `'auto'` literal):**
Chave apagada de `config.BASE_INFORMATION` para as 4 séries de `FS_DEV_SERIES` — `resolve_lag_size()`
cai no fallback (`lag_size`), idêntico ao baseline (paridade de `df_train` provada por teste).
**Valor real que `lag_size='auto'` resolve hoje (medido via PACF, não estimado):**

| Série | `lag_size` auto | ⚠️ |
|---|---|---|
| `airlines.txt` | 20 | |
| `coloradoRiver.txt` | 16 | |
| `sunspot.txt` | 9 | |
| `austres.txt` | **1** | Praticamente sem espaço de seleção de features — avaliar com o pesquisador antes de incluir essa série numa comparação entre métodos de FS. |

**Parte C — Registro de features selecionadas:** `src/utils/export_selected_features.py` (novo)
— extração pós-hoc dos `.pkl` já existentes (zero mudança em `generics.py`), gera
`selected_features.csv`/`_detail.csv` por `experiment_id`.

**Parte D — Comparação lado a lado:** `src/utils/compare_fs_vs_baseline.py` (novo) — baseline
(`data/result/chamados/`, modelo `1amv1`) × cada variante FS, com ganho percentual de RMSE e
nº médio de features. Documenta a ambiguidade histórica de `chamados_v2_fs_ftest/`.

**`experiment_id` novos, propostos e usados nos 4 notebooks: `chamados_v4_fs_{ftest,mutualinfo,
rfembedded,lasso}`** — `chamados_v3_fs_*` fica como registro histórico (nunca reaproveitado).

**Arquivos alterados/novos nesta sessão (working tree, não commitados):**
`src/model/feature_selection.py`, `src/config.py`, `src/utils/export_metrics_to_csv.py`
(extraídas `_load_pkl`/`_unwrap_entry`/`save_csv` compartilhadas), `src/utils/export_selected_features.py`
(novo), `src/utils/compare_fs_vs_baseline.py` (novo), os 4 notebooks em
`notebook/residual_hydridsystem/` (bump v3→v4), `tests/model/test_feature_selection.py`,
`tests/model/test_grid_search_exp.py`, `tests/utils/` (novo dir, 2 arquivos de teste),
`results/chamados_v4_fs_*/metadata.json` (4 novos), `PLANO_ARQUITETURA.md` (Seção 1.5 nova),
`RUNBOOK.md` (Seções 1/2/5/6/7/8b atualizadas).

**Verificação final:** 82/82 testes passando; hashes/mtimes dos 5 baselines em
`data/result/chamados/` confirmados intocados (datados de 2026-03-27, antes desta sessão);
code-review multi-agente (8 ângulos) rodado — 1 bug crítico corrigido (lasso zero-features),
mais 8 achados corrigidos (duplicação de código, validação `k<=0` inconsistente, imports
mortos, `.pkl` do ARIMA colidindo com índice de série em `compare_fs_vs_baseline.py`,
`n_features_in_` ausente em `.pkl` pré-Tarefa-3.1 quebrando extração). Ver Seção 3 para os
achados conscientemente **não** corrigidos (fora de escopo).

**Nenhum `.pkl` novo desta fase foi gerado ainda** — toda execução real continua pendente, por
decisão deliberada de manter a fase manual sob controle do pesquisador (ver RUNBOOK.md Seção 8b
para os comandos prontos).

---

## 2b. O que a Tarefa 3.2 entregou (não commitado ainda)

**Parte A — Fluxo notebook-only (sem terminal, exceto para ativar o ambiente):**
- `src/utils/export_metrics_to_csv.py`/`export_selected_features.py`: extraídas
  `run_export_metrics_to_csv()`/`run_export_selected_features()` de dentro de `main()` — não
  chamam mais `sys.exit()`, então são seguras para chamar direto de uma célula de notebook sem
  matar o kernel. `main()` (CLI) virou um wrapper fino em torno delas, preservando exit
  codes/mensagens idênticos aos de antes.
- `src/utils/copy_pretrained_linear_model.py` (novo, achado de code-review): substitui o loop
  `shutil.copy` que estava duplicado em 4 notebooks — usa `generics.format_names()` (o mesmo
  helper que `Additive`/`input_linear_info` usam pra achar o `.pkl`) e le o nome do modelo
  linear de `experiment_params['linear_model_name']`, em vez de cravar `'1arima'`.
- Os 4 notebooks em `notebook/residual_hydridsystem/` foram reestruturados de 4 para 7 células:
  config consolidada (cell1, com `experiment_dir`/`experiment_dir_results` computados uma vez),
  sanity-check + checagem de identidade `strategy`↔`experiment_id` (cell2, nova — acha
  code-review), cópia do ARIMA via `copy_pretrained_linear_model` (cell3), execução (cell4),
  CSV de métricas (cell5, nova), CSV de features selecionadas (cell6, nova). "Run All" cobre o
  fluxo inteiro.
- `notebook/compare_fs_results.ipynb` (novo): config (baseline_dir + lista de variantes) +
  chamada a `build_comparison()` + `display`/salvamento do CSV, tudo em notebook.
- `RUNBOOK.md` Seção 8b reescrita como fluxo principal notebook-only; comandos de terminal
  preservados nas Seções 3/4/5 como alternativa documentada (não apagados).
- Corrigido de passagem: caracteres Unicode `✓`/`→` nos `print()` de `export_metrics_to_csv.py`/
  `export_selected_features.py` trocados por ASCII (`OK`/`->`) — quebravam com
  `UnicodeEncodeError` em terminal Windows sem `PYTHONIOENCODING=utf-8`.

**Parte B — Investigação do PACF de `austres.txt` (por que `lag_size='auto'` resolve para 1):**
Vetor completo de PACF rodado com o código real do projeto (`src/input/input.py:50-106`,
`get_max_lag_to_consider`, acionado em `src/input/input.py:207`). PACF no lag 1 = **0.973**;
lags 2-20 todos com `|PACF| < 0.035`, dentro do intervalo de confiança 95% (`±0.2178` para
N_treino=81). Conclusão: não é só efeito de amostra pequena — a série é genuinamente dominada
pelo lag 1. Documentado com o comando de reprodução em `RUNBOOK.md` Seção 1.

**Code-review multi-agente (8 ângulos) rodado sobre a Tarefa 3.2** — achados corrigidos:
`try/except FileNotFoundError` amplo demais em ambos os `main()` (podia mascarar um erro não
relacionado vindo de `save_csv` como "diretório não encontrado"), célula de cópia do ARIMA
reimplementando path manualmente com `'1arima'` cravado (→ `copy_pretrained_linear_model`),
falta de checagem que `strategy` e `experiment_id`/`model_name` continuam consistentes entre si
na célula de configuração (→ assert novo na cell2), nota do RUNBOOK.md sobre
`PYTHONIOENCODING`/`✓` desatualizada, import `numpy` morto em teste novo,
`Path(config.MODEL_DATA_PATH) / experiment_id` recalculado 3× por notebook (→ consolidado em
`experiment_dir` na cell1). Achados **não** corrigidos (documentados, fora de escopo): células
de notebook sem try/except ao redor de `run_export_*`/`build_comparison` (comportamento
aceitável em Jupyter — traceback normal, não mata o kernel); risco de reordenar células fora do
"Run All" (ex. editar `fs_series_list` e pular a célula 3) — mitigado pela documentação
explícita do RUNBOOK.md, não por trava de código.

**Verificação final:** 91/91 testes passando (mais 9 desde a Tarefa 3.1: `copy_pretrained_linear_model`
+ `run_export_metrics_to_csv`/`run_export_selected_features`); hashes/mtimes dos 5 baselines
continuam intocados; todos os 5 notebooks (4 FS + `compare_fs_results.ipynb`) validados via
`nbformat`; nenhum notebook foi executado automaticamente nesta sessão.

---

## 2c. O que a Tarefa 3.3 (diagnóstico) e 3.4 entregaram (não commitado ainda)

**Tarefa 3.3 — só investigação, sem código escrito:**
- `notebook/compare_fs_results.ipynb` foi verificado byte-a-byte (JSON bruto) e executado via
  papermill — **não estava corrompido**. O `NameError: name 'null' is not defined` relatado não
  foi reproduzido; hipótese mais provável é um estado transitório no kernel Jupyter (algo colado/
  executado ao vivo, não salvo em disco).
- Confirmado na fonte (`src/model/grid_search_exp.py:70-113`, versão pré-3.4):
  `_search_params()` já calculava, por combinação do grid, a média das repetições internas de
  validação — mas guardava tudo em variáveis locais e retornava só o argmin. Extraí do `.pkl` já
  executado e do `print(best_params)` salvo nas saídas do notebook os vencedores reais
  (`airlines: k=5`, `coloradoRiver: k=9`), mas **não havia como provar** que `k=15`/`k=20` foram
  genuinamente avaliados — essa limitação motivou a Tarefa 3.4.

**Tarefa 3.4 — instrumentação de `GridSearch` (após brainstorming, Seção 5.1 do CLAUDE.md):**
- `GridSearch` ganhou `save_grid_history=True` (padrão) e passa a persistir **todas** as
  combinações testadas — não só a vencedora — numa chave nova `grid_search_history` dentro do
  MESMO `.pkl` (mudança estritamente aditiva; nenhum leitor existente muda de comportamento,
  provado por teste de regressão lendo os `.pkl` reais dos 5 baselines).
- Cada entrada: `{'params', 'val_metric_mean', 'val_metric_std', 'val_metric_reps',
  'val_metrics_reps'}` — o último captura o dict **completo** de métricas de validação por
  repetição (não só RMSE), achado de code-review (já estava em memória, custo zero).
- `load_grid_search_history()` — leitor que vira um DataFrame (uma linha por combinação),
  pronto para `df.plot(x=..., y='val_metric_mean', yerr='val_metric_std')`. Mora em
  `src/model/grid_search_exp.py` (não `src/utils/`), colocado com quem produz o formato —
  ajustado após code-review apontar o precedente de `metrics.open_fold_result`.
- **Bug real encontrado e corrigido pelo code-review**: para `model_class_exp` não-sklearn
  (LSTM/NBEATS/ELM/SCN/etc., via `neural_forecast_exp.py`/`perturbative_neural_forecast.py`),
  o dict de hiperparâmetros é mutado in-place pelo wrapper (`random_seed`, `input_size`, etc.
  injetados) — sem a correção, o histórico persistido capturaria essa poluição em vez dos
  hiperparâmetros reais do grid. Não afeta os 5 baselines nem os notebooks de FS (usam
  `Pipeline` sklearn, imunes a essa mutação), mas era um bug genuíno na feature nova.
- **Limitação aceita, não contornada**: o histórico de `chamados_v4_fs_ftest` (rodado antes
  desta instrumentação) não é recuperável — só existiu como variável local numa execução já
  finalizada. Vale só para rodadas futuras.

**Verificação final:** 106/106 testes passando; hashes/mtimes dos 5 baselines continuam
intocados; prova de conceito real (não sintética) rodada — grid `selector__k=[1,3,5,8]` em
`airlines.txt`/`f_test` via `Additive` real, produzindo o gráfico clássico "erro de validação
vs. k" com barras de erro. Nenhum experimento real (`chamados_v4_fs_*`) foi re-executado.

---

## 3. Pendências conhecidas

0. **⚠️ PROVISÓRIA (Tarefa 6/7) — `gamma='auto'` do `SVR` confunde efeito de FS com largura
   de kernel.** `gamma='auto'` = `1/n_features`, então reduzir features via FS muda o kernel
   automaticamente junto com a seleção. Decisão do pesquisador: manter `gamma='auto'` por
   ora, a revisar com o orientador (ver PLANO_ARQUITETURA.md Seção 1.11 para o detalhe
   completo). **Se a decisão mudar, as famílias `SVR` single e `ARIMA-SVR` precisam ser
   re-rodadas** — diferente das outras pendências desta lista, esta não é um bug a corrigir,
   é uma decisão de metodologia em aberto.
1. ~~`activation='logistic'` em `arima_mlp.ipynb` nunca foi formalmente documentado no
   CLAUDE.md.~~ **Resolvido na Tarefa 3.9** — formalizado no CLAUDE.md Seção 3.6.
1b. **Execução manual pendente (Tarefa 3.9): 3 notebooks preparados, nenhum rodado ainda.**
   `arima_mlp.ipynb` (força regeneração das 17 séries de `1amv1`, `activation='logistic'`),
   `arima_mlp_rf_embedded.ipynb` e `arima_mlp_lasso.ipynb` (grid limpo, sem `selector__k`
   morto) estão prontos com `force=True`, mas precisam ser rodados via "Run All" pelo
   pesquisador (ver relatório da Tarefa 3.9 para o passo a passo). Depois:
   `compare_fs_results.ipynb` e a análise da Tarefa 3.7 (`results/chamados_v4_fs_analysis_report.md`)
   precisam ser refeitos com os dados novos.
1c. ~~Divergência do `1mlp`/`diff_kpss` (descoberta na Tarefa 3.9).~~ **Resolvido — `1mlp`
   regenerado e confirmado (Tarefas 5.1/5.2)**, `diff_kpss=False` nos 17 `.pkl`. Versão antiga
   arquivada em `data/result/chamados_baseline_mlp_diffkpss_true_archive_20260721/`.
2. **`grid_seach_multiple_bases`/`GridSearch` (`src/model/grid_search_exp.py`) não têm
   try/except por série.** Se uma série falhar no meio de um laço sobre `BASE_NAME_LIST`, as
   séries seguintes são silenciosamente puladas, sem registro de erro (achado do code-review
   da Tarefa 3.1, fora de escopo — tocar `grid_search_exp.py` estruturalmente exige
   `brainstorming` prévio pela Seção 5.1 do CLAUDE.md). Vale considerar antes da Tarefa 4.
3. **`export_selected_features.py` depende de um detalhe de implementação incidental**
   (`Additive`/`SKlearnModel` preservarem `self.model` como o Pipeline ajustado após o fit) —
   decisão de design deliberada e aprovada nesta sessão (ver PLANO_ARQUITETURA.md Seção 1.5),
   não um bug, mas um acoplamento a documentar se `hybrid_system_exp.py`/`generics.py` forem
   refatorados no futuro.
4. **`TimeSeriesFeatureSelector.fit()` acumula 4 branches if/elif** misturando duas famílias de
   estratégia (filtro top-k vs. `SelectFromModel` embedded). Funcional hoje, mas o roadmap da
   Tarefa 4 (`rfecv`, `perm_importance`) empurraria para 6 branches — vale considerar despacho
   por dicionário/classes de estratégia antes de adicionar os próximos 2 métodos.
5. Item 9 do relatório da Tarefa 2.6 (dois notebooks voltaram sozinhos ao HEAD) — sem ação
   pendente, só histórico.
6. **Risco de reordenar células fora do "Run All" nos 4 notebooks de FS (achado da Tarefa 3.2,
   não corrigido).** Editar `fs_series_list` na célula 1 e re-rodar só a célula 4 (pulando a 3)
   deixa o `.pkl` do ARIMA da série nova sem copiar — `GridSearch`/`Additive` levanta
   `FileNotFoundError` sem tratamento no meio do laço. Mitigado hoje só por documentação
   (RUNBOOK.md Seção 8b recomenda sempre "Run All"), não por trava de código.

---

## 4. Documentos fundamentais (ordem de leitura recomendada)

1. `CLAUDE.md` — regras imutáveis do repositório (leitura obrigatória, sempre).
2. `PLANO_ARQUITETURA.md` — arquitetura da fase de Feature Selection, arsenal de métodos,
   roadmap (Seção 3), convenção de nomenclatura de notebooks (Seção 5) e a decisão de design
   da Tarefa 3.1 (Seção 1.5).
3. `RUNBOOK.md` — comandos operacionais para rodar/repetir experimentos (Seção 8b = rodada atual).
4. Este arquivo (`CHECKPOINTS.md`) — estado atual, sempre a fonte mais recente de "onde paramos".

---

## 5. Próximos passos sugeridos

1. Revisar este arquivo e o relatório da Tarefa 3.2, decidir se commita o working tree atual
   (Tarefas 3.1 + 3.2 juntas, ou separadas — decisão sua).
2. Se aprovado: primeira execução manual real. Fluxo principal agora é abrir cada um dos 4
   notebooks em `notebook/residual_hydridsystem/` e rodar "Run All" (RUNBOOK.md Seção 8b) — a
   única edição normalmente necessária é a célula 1 (configuração), já pré-preenchida com
   `experiment_id=chamados_v4_fs_*`. Nenhuma execução foi feita automaticamente nesta sessão.
3. Prestar atenção especial ao resultado de `austres.txt` (1 lag candidato, PACF concentrada no
   lag 1 — ver Seção 2b) antes de comparar métodos de FS entre si.
4. Depois dos 4 rodarem: abrir `notebook/compare_fs_results.ipynb` e "Run All" — gera
   `results/chamados_v4_fs_comparison.csv` consolidando o comparativo.
5. Tarefa 4 (`rfecv`/`perm_importance`) — considerar a pendência #4 da Seção 3 (despacho por
   estratégia em vez de if/elif) antes de começar, já que essa tarefa adicionaria mais 2 branches.

Não deixar de **atualizar este arquivo e commitar** ao final de cada tarefa daqui em diante.
