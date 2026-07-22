# RUNBOOK.md — Rodando Experimentos de Feature Selection

Guia prático para você (pesquisador) repetir/variar experimentos de Feature Selection sem reconstruir o raciocínio do zero. Para as regras que **não** podem mudar, veja [CLAUDE.md](CLAUDE.md); para o desenho da arquitetura, [PLANO_ARQUITETURA.md](PLANO_ARQUITETURA.md).

**Desde a Tarefa 3.2, o fluxo principal é notebook-only** (abrir o notebook, editar a célula de configuração, "Run All" — sem terminal, exceto para ativar o ambiente): ver Seção 8b. As Seções 1-7 abaixo explicam o que cada campo da célula de configuração significa; os comandos de terminal equivalentes continuam documentados como alternativa (Seções 4, 5 e a rodada histórica da Seção 8) para quem preferir esse caminho.

---

## 1. Trocar a série

Hoje restrito a `FS_DEV_SERIES` (`airlines`, `austres`, `coloradoRiver`, `sunspot` — definido em [tests/model/conftest.py](tests/model/conftest.py)). Expandir a lista além dessas 4 é decisão explícita futura, não algo a fazer implicitamente ao seguir este runbook.

Desde a Tarefa 3.2, a série entra na **célula de configuração** consolidada no topo do notebook (célula 1, junto de `experiment_id`/`model_name`/grid — ver Seção 8b):

```python
fs_series_list = ['airlines.txt', 'austres.txt', 'coloradoRiver.txt', 'sunspot.txt']   # troque/reduza aqui
```

**Desde a Tarefa 3.1 (PLANO_ARQUITETURA.md Seção 1.5), `fs_lag_size` foi REMOVIDA de `config.py`** para as 4 séries (reversão pedida pelo orientador — não mais um valor profundo manual por série). `resolve_lag_size()` cai no fallback (`lag_size='auto'`), **idêntico ao baseline** — provado por `tests/model/test_grid_search_exp.py::TestFsDevSeriesDfTrainParity` (paridade real de `df_train`, não apenas do valor resolvido).

O valor real que `lag_size='auto'` resolve hoje (via PACF, medido com o pipeline real — pré-check da Tarefa 3.1, não estimado):

| Série | `lag_size` resolvido (`auto`) | N total | Observação |
|---|---|---|---|
| `airlines.txt` | 20 | 144 | |
| `coloradoRiver.txt` | 16 | 744 | |
| `sunspot.txt` | 9 | 288 | |
| `austres.txt` | **1** | 89 | ⚠️ **Risco à narrativa de comparação entre métodos de FS**: com apenas 1 lag candidato, não há praticamente nenhum espaço real de seleção de features para `austres.txt` — qualquer estratégia (f_test/mutual_info/rf_embedded/lasso) tende a "selecionar" a única feature disponível. Avaliar com o pesquisador se `austres.txt` deve ficar de fora da comparação de FS, ou se o resultado (variância zero entre métodos) é, em si, um dado a discutir. |

Esses valores **não são mais uma chave gravada em `config.py`** (evita reintroduzir um terceiro esquema de configuração, CLAUDE.md Seção 3.1) — estão documentados aqui e em `results/chamados_v4_fs_*/metadata.json` (`lag_size_auto_resolved`) apenas como referência.

**Por que `austres.txt` resolve para 1 (investigação da Tarefa 3.2):** `austres.txt` (série trimestral, N=89) tem PACF no lag 1 = **0.973** — praticamente toda a autocorrelação parcial da série está concentrada aí — e todos os lags 2 a 20 ficam com `|PACF| < 0.035`, bem dentro do intervalo de confiança 95% (`±1.96/√N_treino = ±0.2178`). Não é só efeito de amostra pequena "engolindo" sinal real: a série é genuinamente dominada pelo lag 1, e o resto é ruído estatístico corretamente identificado como não-significativo pela lógica em [src/input/input.py:50-106](src/input/input.py#L50-L106) (`get_max_lag_to_consider`), acionada em [src/input/input.py:207](src/input/input.py#L207) quando `lag_size='auto'`. Para reproduzir o vetor completo de PACF:

```bash
.venv/Scripts/python.exe -c "
from statsmodels.tsa.stattools import pacf
import numpy as np, config
from input.input import load_raw_data
ts = load_raw_data('austres.txt')['y'].values
train = ts[:-int(config.TEST_SIZE * len(ts))]
lag_pacf = pacf(train, nlags=20)
limit = 1.96 / np.sqrt(len(train))
for lag, p in enumerate(lag_pacf):
    print(lag, round(p, 4), 'SIGNIFICATIVO' if abs(p) > limit else '')
"
```

## 2. Trocar a estratégia de FS e os hiperparâmetros do seletor

Desde a Tarefa 3.2, `model` e `model_parameters` vivem na mesma célula de configuração da Seção 1 (célula 1 do notebook). Estratégia (fixa por notebook — nunca misturar no mesmo grid):

```python
model = Pipeline([
    ('selector', TimeSeriesFeatureSelector(strategy='mutual_info')),  # 'f_test' | 'mutual_info' | 'rf_embedded' | 'lasso' hoje
    ('estimator', MLPRegressor(activation='logistic', solver='lbfgs')),
])
```

`rf_embedded` (`RandomForestRegressor`, importância por redução de impureza) e `lasso` (`LassoCV(cv=TimeSeriesSplit(n_splits=3))`, regularização L1) foram adicionados na Tarefa 3. `rfecv` (`RFECV(RandomForestRegressor, cv=TimeSeriesSplit(n_splits=3), min_features_to_select=1)`) foi adicionado na Tarefa 4 — implementado e testado, notebook (`arima_mlp_rfecv.ipynb`) pronto mas **ainda não executado**. `perm_importance` continua não implementado (stretch opcional, fora do escopo obrigatório).

**Desde a Tarefa 3.1 (PLANO_ARQUITETURA.md Seção 1.5), `f_test`/`mutual_info` e `rf_embedded`/`lasso` seguem contratos diferentes de hiperparâmetro:**

```python
# f_test / mutual_info -- k continua um hiperparametro pesquisavel (SelectKBest-style)
model_parameters = {
    'selector__k': [5, 10, 15],
    'estimator__hidden_layer_sizes': [10, 20, 50],
    'estimator__max_iter': [1000],
}

# rf_embedded / lasso -- k foi REMOVIDO. SelectFromModel(threshold=None) decide o
# numero de features pelo proprio ajuste (sklearn resolve threshold=None para 'mean'
# das importancias no RF, e para ~1e-5 no Lasso via deteccao de L1) -- nao ha mais
# `selector__k` para passar no grid.
model_parameters = {
    'estimator__hidden_layer_sizes': [10, 20, 50],
    'estimator__max_iter': [1000],
}
```

O número de features selecionadas por `rf_embedded`/`lasso` passa a **variar por execução** (inclusive entre as `model_exec=10` repetições, para `rf_embedded` — RF usa `random_state=None` deliberadamente, para observar a variabilidade real). Isso fica registrado por série/repetição em `results/<experiment_id>/selected_features_detail.csv` (ver Seção 5 abaixo).

## 2b. Histórico completo do Grid Search — erro de validação por combinação (Tarefa 3.4)

Desde a Tarefa 3.4 (PLANO_ARQUITETURA.md Seção 1.6), `GridSearch` persiste **todas** as combinações testadas — não só a vencedora — numa chave nova e aditiva `grid_search_history`, dentro do mesmo `.pkl` de sempre (nenhum leitor existente muda de comportamento). Isso vale automaticamente para toda execução futura (`save_grid_history=True` por padrão); **não é retroativo** — `.pkl` gerados antes da Tarefa 3.4 (ex. `chamados_v4_fs_ftest`, achado da Tarefa 3.3) não têm essa chave e não são recuperáveis.

Leitura (uma linha por combinação, pronta para plotar "erro de validação vs. hiperparâmetro"):

```python
from pathlib import Path
from model.grid_search_exp import load_grid_search_history

df = load_grid_search_history(Path('data/result/<experiment_id>/<serie>_<horizon><model_name>.pkl'))
df = df.sort_values('selector__k')  # ou o hiperparametro que te interessa
print(df[['selector__k', 'val_metric_mean', 'val_metric_std']])

df.plot(x='selector__k', y='val_metric_mean', yerr='val_metric_std', marker='o', capsize=3)
```

Cada linha tem uma coluna por chave de `params` (ex. `selector__k`, `estimator__hidden_layer_sizes`) mais `val_metric_mean`/`val_metric_std`/`val_metric_reps` (as `model_exec` repetições internas cruas de RMSE daquela combinação, nunca métrica de teste — `GridSearch.group_metrics_name` é fixado como `'val_metrics'`) e `val_metrics_reps` (lista com o dict **completo** de validação por repetição — `MSE`/`RMSE`/`MAE`/`MAPE`/`theil`/`ARV`/`IA`/`POCID` — para quem quiser plotar por outra métrica além de RMSE, sem precisar re-rodar o grid). Exemplo real (grid `selector__k=[1,3,5,8]`, `airlines.txt`, `f_test`, rodada de demonstração — não um experimento oficial):

| `selector__k` | `val_metric_mean` | `val_metric_std` |
|---|---|---|
| 1 | 14.99 | 0.0001 |
| 3 | 14.30 | 0.0067 |
| 5 | 14.34 | 0.0045 |
| 8 | 14.54 | 0.0060 |

Se `load_grid_search_history` levantar `ValueError`, o `.pkl` não tem o histórico (gerado com `save_grid_history=False`, antes da Tarefa 3.4, ou com `grid_search_history` presente mas vazio) — re-rode o Grid Search para capturá-lo.

## 3. Definir um novo `experiment_id`

Regra (CLAUDE.md, Seção 3.2): **nome novo e explícito, nunca reaproveitar um `experiment_id` existente para uma configuração diferente.** Sem hash automático, sem validação de "config mudou" — a responsabilidade é sua.

Convenção sugerida: `chamados_v<N>_fs_<estrategia>` (ex.: `chamados_v3_fs_mutualinfo`, `chamados_v4_fs_rfembedded`).

Onde isso aparece:

| Caminho | O que fica lá |
|---|---|
| `data/result/<experiment_id>/` | `.pkl` de cada série × modelo (gerado pelo `GridSearch`) |
| `results/<experiment_id>/metrics.csv` | Agregado (média das repetições), via `export_metrics_to_csv.py` |
| `results/<experiment_id>/metrics_detail.csv` | Detalhado (uma linha por repetição) |
| `notebook/executed/<nome_do_notebook>.ipynb` | Notebook executado (papermill) — **nunca** aponte para `notebook/executed/arima_mlp.ipynb`, que é o registro do baseline original |

Se o híbrido depende de ARIMA pré-treinado (`Additive`, via `experiment_params['linear_model_name']`), o `.pkl` do ARIMA precisa existir **sob o mesmo `experiment_id` novo**. Como o ARIMA em si não muda com `fs_lag_size`/estratégia de FS, copie (não retreine) do experimento que já tem esse ARIMA. Desde a Tarefa 3.2, isso é feito automaticamente pela célula 3 do notebook (`shutil.copy`, ver Seção 8b) — o equivalente em terminal, para quem preferir:

```bash
mkdir -p data/result/<novo_experiment_id>
cp data/result/chamados/<serie>_1arima.pkl data/result/<novo_experiment_id>/<serie>_1arima.pkl
```

## 4. Rodar o notebook via papermill (alternativa via terminal)

Fluxo principal desde a Tarefa 3.2 é abrir o notebook e rodar "Run All" (Seção 8b). Este comando é a alternativa via terminal — útil para automação/CI, sem precisar abrir o Jupyter. Comando real, confirmado na Tarefa 2 (roda a partir da raiz do projeto, `.venv` ativo):

```bash
export PYTHONPATH="$(pwd)/src:$(pwd):$PYTHONPATH"
.venv/Scripts/python.exe -m papermill \
  notebook/residual_hydridsystem/<notebook_da_combinacao>.ipynb \
  notebook/executed/<notebook_da_combinacao>_<sufixo_da_rodada>.ipynb \
  --kernel python3 \
  --execution-timeout 3600 \
  --progress-bar
```

O `PYTHONPATH` é obrigatório — sem ele, `import config`/`from model import ...` falham dentro do kernel do papermill.

## 5. Agregar métricas e exportar CSV (alternativa via terminal)

Desde a Tarefa 3.2, as células 5-6 de cada notebook de FS (e o notebook `notebook/compare_fs_results.ipynb`) já geram esses CSVs automaticamente via "Run All" (Seção 8b), chamando as mesmas funções Python descritas abaixo — sem precisar do terminal. Os comandos desta seção continuam funcionando, para quem preferir esse caminho ou quiser rodar so uma etapa isolada.

`calculate_metrics_v2.ipynb` **não** tem célula `parameters` (papermill não consegue injetar `experiment_id` nele) e as células 5-9 têm filtros de nome de modelo hardcoded da nomenclatura antiga dos 5 baselines — não funcionam para um `experiment_id` novo. Use estes dois caminhos em vez disso:

**Leitura agregada (equivalente à célula 2 do notebook, sem as partes quebradas):**

```bash
.venv/Scripts/python.exe -c "
from metrics.metrics import open_fold_result
df_mean, df_all, df_prevs = open_fold_result('<experiment_id>', 'val_metrics', 'RMSE')
print(df_mean)
"
```

**Exportação para CSV, isolada por etapa (já suporta `--result-dir`/`--output`, sem precisar editar o script):**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/utils/export_metrics_to_csv.py \
  --result-dir data/result/<experiment_id> \
  --output results/<experiment_id>/metrics.csv \
  --detail
```

`PYTHONIOENCODING=utf-8` evita um erro de encoding do console Windows ao imprimir acentos ("média", "repetições" etc.) em terminais fora do codepage `cp1252` — não é um problema do script, só do terminal. (Antes da Tarefa 3.2, o script também imprimia `✓`/`→`, que quebravam mesmo em `cp1252`; esses dois caracteres foram trocados por ASCII simples — achado de code-review — então hoje só resta esse risco cosmético de acentuação, não mais um crash.)

**Nunca** rode `export_metrics_to_csv.py` sem `--result-dir`/`--output` explícitos para um experimento novo — o padrão sem esses parâmetros escreve em `results/baseline_metrics.csv`, sobrescrevendo o relatório dos 5 baselines originais.

**Registro de features selecionadas (Tarefa 3.1, PLANO_ARQUITETURA.md Seção 1.5) — `src/utils/export_selected_features.py`:**

Extração pós-hoc, mesmo padrão de `export_metrics_to_csv.py` — lê o Pipeline/seletor já ajustado que sobrevive dentro do `.pkl` (sem precisar de nenhuma mudança em `generics.py`). `.pkl` sem seletor (os 5 baselines) são ignorados com aviso, não erro:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/utils/export_selected_features.py \
  --result-dir data/result/<experiment_id> \
  --output results/<experiment_id>/selected_features.csv \
  --detail
```

Gera `selected_features.csv` (agregado: média/desvio de `N_Features_Selected` por série×modelo×estratégia) e `selected_features_detail.csv` (uma linha por repetição, com `Selected_Indices` e `Selected_Lag_Names` — ex. `lag_20;lag_15;lag_3`). Para `rf_embedded`, compare o desvio-padrão entre repetições no detail: variância real (RF sem seed fixo) é o dado esperado, não um bug.

**Comparação lado a lado baseline × variantes FS — `src/utils/compare_fs_vs_baseline.py`:**

Reaproveita `export_metrics_to_csv`/`export_selected_features` sobre os mesmos `.pkl` — não é uma terceira extração. `--baseline-dir` **sempre** aponta para `data/result/chamados` (os 5 baselines intocáveis); **nunca** para uma pasta `chamados_v*_fs_*` — ver a nota "AMBIGUIDADE HISTÓRICA" no topo do próprio script sobre `chamados_v2_fs_ftest` (lá, `1amv1` já é a variante f_test, não o baseline):

```bash
.venv/Scripts/python.exe src/utils/compare_fs_vs_baseline.py \
  --baseline-dir data/result/chamados \
  --fs ftest:data/result/chamados_v4_fs_ftest \
  --fs mutualinfo:data/result/chamados_v4_fs_mutualinfo \
  --fs rfembedded:data/result/chamados_v4_fs_rfembedded \
  --fs lasso:data/result/chamados_v4_fs_lasso \
  --output results/chamados_v4_fs_comparison.csv
```

Gera uma linha por série, com `Baseline_RMSE` e, para cada rótulo em `--fs`: `<rotulo>_RMSE`, `<rotulo>_PctGain` (positivo = variante melhor que o baseline) e `<rotulo>_NFeatures` (média de features selecionadas, quando `selected_features` estiver disponível para aquele `.pkl`). Série sem `.pkl` para uma variante específica vira `NaN`, não quebra a comparação inteira. Equivalente notebook: `notebook/compare_fs_results.ipynb` (Seção 8b).

## 6. Convenção de nomenclatura de notebooks (FS × arquitetura)

Detalhada na Seção 5 do [PLANO_ARQUITETURA.md](PLANO_ARQUITETURA.md#5-convenção-de-nomenclatura-de-notebooks-por-combinação-fs--arquitetura). Resumo:

```
<hibrido>_<estrategia_fs>.ipynb          (notebook/residual_hydridsystem/)
<single_model>_<estrategia_fs>.ipynb     (notebook/single_models/ -- Tarefa 5)
```

| Notebook | Status | `experiment_id` proposto |
|---|---|---|
| `arima_mlp_ftest.ipynb` | **Executado** (Tarefa 3.2; baseline de comparação corrigido na Tarefa 3.9 — `.pkl` deste notebook em si não precisou regenerar) | `chamados_v4_fs_ftest` |
| `arima_mlp_mutual_info.ipynb` | **Executado** (Tarefa 3.2; idem acima) | `chamados_v4_fs_mutualinfo` |
| `arima_mlp_rf_embedded.ipynb` | **Executado e regenerado** (Tarefa 3.9: `selector__k` morto removido, grid limpo) | `chamados_v4_fs_rfembedded` |
| `arima_mlp_lasso.ipynb` | **Executado e regenerado** (Tarefa 3.9: `selector__k` morto removido, grid limpo) | `chamados_v4_fs_lasso` |
| `arima_mlp_rfecv.ipynb` | **Pronto, NÃO executado** (Tarefa 4: `strategy='rfecv'` implementado em `feature_selection.py`, TDD completo, custo estimado ~13-14min para as 4 séries — ver célula de configuração do notebook) | `chamados_v4_fs_rfecv` |
| `mlp_ftest.ipynb` | **Executado** (Tarefa 5 — ver Seção 6b abaixo) | `chamados_v4_fs_mlp_ftest` |
| `mlp_mutual_info.ipynb` | **Executado** (Tarefa 5) | `chamados_v4_fs_mlp_mutualinfo` |
| `mlp_rf_embedded.ipynb` | **Executado** (Tarefa 5) | `chamados_v4_fs_mlp_rfembedded` |
| `mlp_lasso.ipynb` | **Executado** (Tarefa 5) | `chamados_v4_fs_mlp_lasso` |
| `mlp_rfecv.ipynb` | **Executado** (Tarefa 5) | `chamados_v4_fs_mlp_rfecv` |
| `svr_ftest.ipynb` | **Executado** (Tarefa 6 — ver Seção 6c abaixo) | `chamados_v4_fs_svr_ftest` |
| `svr_mutual_info.ipynb` | **Executado** (Tarefa 6) | `chamados_v4_fs_svr_mutualinfo` |
| `svr_rf_embedded.ipynb` | **Executado** (Tarefa 6) | `chamados_v4_fs_svr_rfembedded` |
| `svr_lasso.ipynb` | **Executado** (Tarefa 6) | `chamados_v4_fs_svr_lasso` |
| `svr_rfecv.ipynb` | **Executado** (Tarefa 6) | `chamados_v4_fs_svr_rfecv` |
| `arima_svr_ftest.ipynb` | **Pronto, NÃO executado** (Tarefa 7 — ver Seção 6d abaixo) | `chamados_v4_fs_arimasvr_ftest` |
| `arima_svr_mutual_info.ipynb` | **Pronto, NÃO executado** (Tarefa 7) | `chamados_v4_fs_arimasvr_mutualinfo` |
| `arima_svr_rf_embedded.ipynb` | **Pronto, NÃO executado** (Tarefa 7) | `chamados_v4_fs_arimasvr_rfembedded` |
| `arima_svr_lasso.ipynb` | **Pronto, NÃO executado** (Tarefa 7) | `chamados_v4_fs_arimasvr_lasso` |
| `arima_svr_rfecv.ipynb` | **Pronto, NÃO executado** (Tarefa 7) | `chamados_v4_fs_arimasvr_rfecv` |
| `nonlinear_mlp_ftest.ipynb` (e variantes) | Não implementado — só quando o roadmap de combinações não-lineares de 2–3 estágios (CLAUDE.md Seção 4) for implementado |  |

`chamados_v3_fs_*` (Tarefa 3) está **superado** pela Tarefa 3.1 — os 4 notebooks usavam top-k uniforme para `rf_embedded`/`lasso` e `fs_lag_size` manual, nunca chegaram a ser executados (sem `.pkl`/métricas gerados). Os templates `results/chamados_v3_fs_*/metadata.json` ficam como registro histórico; não copie deles para uma rodada nova — use os `results/chamados_v4_fs_*/metadata.json`.

`arima_mlp.ipynb`/`arima_svr.ipynb`/`mlp_exec.ipynb`/`svr_exec.ipynb` (sem sufixo) **nunca** recebem essas mudanças — são os baselines intocáveis da Seção 3 do CLAUDE.md.

### 6b. `MLP` single (Tarefa 5) — primeira família fora de `Additive`

Os 5 notebooks `notebook/single_models/mlp_<estrategia>.ipynb` seguem o mesmo padrão de 6 células dos notebooks de `ARIMA-MLP` (config única, sanity-check, execução via `GridSearch(...).execution()` direto por série, CSV de métricas, CSV de features) — **exceto** a célula de "copiar o ARIMA pré-treinado", que não existe aqui: `SKlearnModel` opera direto sobre a série bruta, sem depender de nenhum modelo linear.

Validado (Tarefa 5, sem gerar `.pkl` de produção):
- `SKlearnModel` é agnóstico à identidade de `model` (`Pipeline` funciona sem mudança de código) — teste de integração real em `tests/model/test_single_ml_model_exp.py`.
- `lag_size='auto'` mede os MESMOS valores que o híbrido para as 4 séries de `FS_DEV_SERIES`: `airlines`=20, `austres`=1, `coloradoRiver`=16, `sunspot`=9. `df_train` tem mais linhas que o híbrido (a série bruta é janelada uma única vez; o resíduo do ARIMA é janelado duas vezes) — diferença estrutural esperada, não um problema.
- As 5 células de config/sanity-check de cada notebook foram executadas de verdade (não só validadas como JSON) contra o pipeline real, confirmando ausência de erro de sintaxe/lógica antes de entregar.

Executados manualmente pelo pesquisador (Tarefa 5) — validação estrutural pós-execução (20 `.pkl`, `activation`/`diff_kpss` corretos, sem contaminação da família híbrida) confirmada num portão dedicado antes da Tarefa 6. `results/chamados_v4_fs_mlp_comparison.csv` + `notebook/compare_fs_results_mlp.ipynb` geram a comparação rápida contra o baseline `1mlp`.

### 6c. `SVR` single (Tarefa 6) — mesmo padrão da 6b, com 2 diferenças reais

Os 5 notebooks `notebook/single_models/svr_<estrategia>.ipynb` seguem exatamente a mesma estrutura de 6 células da Seção 6b (`MLP` single) — a integração `Pipeline([selector, estimador])` já validada generaliza sem mudança de código (confirmado com teste real, `TestSKlearnModelAcceptsPipelineWithSVR`). Duas diferenças reais em relação ao template de MLP (não um copy-paste ingênuo):

- **`model_exec=1`** (SVR é determinístico, CLAUDE.md Seção 3.4 — mesma convenção do baseline `svr_exec.ipynb`), não 10.
- **`diff_kpss=True`** (config real do baseline `svr_exec.ipynb`, já confirmada sem divergência de artefato na Tarefa 3.9). `lag_size='auto'` mede os MESMOS valores já conhecidos (`airlines`=20, `austres`=1, `coloradoRiver`=16, `sunspot`=9) — confirmado com `diff_kpss=True` de verdade, não assumido do valor de `MLP` (`diff_kpss=False`). `df_train` tem 1 linha a menos que o `MLP` single para a mesma série (diferenciação KPSS consome 1 linha) — diferença estrutural esperada.

Grid de hiperparâmetros extraído de `svr_exec.ipynb`: `C=[10,100,1000]`, `gamma=['auto']`, `kernel=['rbf']`, `epsilon=[0.1,0.01,0.001]`, `tol=[0.001]` (prefixo `estimator__` nos 5 notebooks novos).

Executados manualmente pelo pesquisador (Tarefa 6) — validação estrutural pós-execução confirmada num portão dedicado antes da Tarefa 7: `svr_exec.ipynb` (baseline) rodou como no-op seguro (`force=False`, sem divergência conhecida — mtime idêntico ao pré-sessão); grid de hiperparâmetros dos 20 `.pkl` de FS com paridade total ao baseline. `results/chamados_v4_fs_svr_comparison.csv` + `notebook/compare_fs_results_svr.ipynb` geram a comparação rápida contra o baseline `1svr`.

### 6d. `ARIMA-SVR` (Tarefa 7) — última família da matriz, combina 6b/6c com o padrão híbrido

Os 5 notebooks `notebook/residual_hydridsystem/arima_svr_<estrategia>.ipynb` combinam o padrão híbrido de 7 células (config, sanity-check, **cópia do ARIMA pré-treinado**, execução, CSV de métricas, CSV de features — igual a `arima_mlp_<estrategia>.ipynb`) com o estimador `SVR` (mesmas 2 diferenças reais da Seção 6c: `model_exec=1`, determinístico; grid `C`/`gamma`/`kernel`/`epsilon`/`tol` extraído de `arima_svr.ipynb`). `diff_kpss=False` aqui — **diferente** de `svr_exec.ipynb`/Seção 6c (`diff_kpss=True`), igual a `arima_svr.ipynb` (já confirmado sem divergência de artefato na Tarefa 3.9).

`lag_size='auto'` medido diretamente no contexto híbrido (resíduo do ARIMA, `diff_kpss=False`): `airlines`=20, `austres`=1, `coloradoRiver`=16, `sunspot`=9 — idêntico ao ARIMA-MLP, porque o resíduo em si não depende do estimador a jusante (MLP ou SVR), confirmado com teste real (`tests/model/test_hybrid_system_exp.py`), não assumido.

**⚠️ Nota provisória (não definitiva, ver PLANO_ARQUITETURA.md Seção 1.11):** `gamma='auto'` do `SVR` é `1/n_features` — muda junto com a seleção de features, confundindo o efeito de FS com o efeito de largura de kernel. Decisão do pesquisador: manter `gamma='auto'` por ora (não fixar), a revisar com o orientador. Se mudar, esta família e a `SVR` single (Seção 6c) precisam ser re-rodadas.

**Nenhum dos 5 notebooks foi executado.**

## 7. Nomenclatura do `.pkl` com seletor e metadado da rodada

Notebooks com `Pipeline`/seletor produzem `.pkl` com sufixo de estratégia **sem underscore**, concatenado direto ao `model_name` do híbrido (ex. `1amv1ftest`, `1amv1mutualinfo`, `1amv1rfembedded`, `1amv1lasso`) — nunca `1amv1_ftest` (underscore quebraria a extração de nome de modelo em `metrics.py`, que pega só o último token separado por `_`; provado por teste de integração em `tests/metrics/test_metrics.py`). Os 5 baselines continuam sem qualquer sufixo.

Cada `results/<experiment_id>/` tem um `metadata.json` documentando estratégia, grids de hiperparâmetro (ou mecanismo de `SelectFromModel`, para `rf_embedded`/`lasso`), séries incluídas e `lag_size_auto_resolved` por série (referência apenas — não é mais uma chave de `config.py`, ver Seção 1) — evita ter que adivinhar isso olhando só o CSV. Os 4 templates da Tarefa 3.1 já existem (`results/chamados_v4_fs_*/metadata.json`); ao rodar uma configuração nova, copie um desses como ponto de partida.

## 8. Comandos da rodada anterior (Tarefa 3 — superada, nunca executada)

**Histórico, não executar.** Os comandos abaixo (`chamados_v3_fs_*`) refletem a configuração da Tarefa 3 (top-k uniforme para `rf_embedded`/`lasso`, `fs_lag_size` manual) — revertida na Tarefa 3.1. Mantidos aqui só como registro; a rodada atual é a Seção 8b.

**Passo 0 — copiar o ARIMA pré-treinado para as 4 pastas novas (mesmo modelo, não retreina):**

```bash
for exp in chamados_v3_fs_ftest chamados_v3_fs_mutualinfo chamados_v3_fs_rfembedded chamados_v3_fs_lasso; do
  mkdir -p "data/result/$exp"
  for serie in airlines austres coloradoRiver sunspot; do
    cp "data/result/chamados/${serie}_1arima.pkl" "data/result/$exp/${serie}_1arima.pkl"
  done
done
```

**Passo 1 — rodar cada notebook via papermill, um de cada vez (observe a saída antes de ir para o próximo):**

```bash
export PYTHONPATH="$(pwd)/src:$(pwd):$PYTHONPATH"

.venv/Scripts/python.exe -m papermill \
  notebook/residual_hydridsystem/arima_mlp_ftest.ipynb \
  notebook/executed/arima_mlp_ftest_v3.ipynb \
  --kernel python3 --execution-timeout 3600 --progress-bar

.venv/Scripts/python.exe -m papermill \
  notebook/residual_hydridsystem/arima_mlp_mutual_info.ipynb \
  notebook/executed/arima_mlp_mutual_info_v3.ipynb \
  --kernel python3 --execution-timeout 3600 --progress-bar

.venv/Scripts/python.exe -m papermill \
  notebook/residual_hydridsystem/arima_mlp_rf_embedded.ipynb \
  notebook/executed/arima_mlp_rf_embedded_v3.ipynb \
  --kernel python3 --execution-timeout 3600 --progress-bar

.venv/Scripts/python.exe -m papermill \
  notebook/residual_hydridsystem/arima_mlp_lasso.ipynb \
  notebook/executed/arima_mlp_lasso_v3.ipynb \
  --kernel python3 --execution-timeout 3600 --progress-bar
```

**Passo 2 — agregar e exportar métricas de cada um (isolado, não toca `results/baseline_metrics.csv`):**

```bash
for exp in chamados_v3_fs_ftest chamados_v3_fs_mutualinfo chamados_v3_fs_rfembedded chamados_v3_fs_lasso; do
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/utils/export_metrics_to_csv.py \
    --result-dir "data/result/$exp" \
    --output "results/$exp/metrics.csv" \
    --detail
done
```

**Passo 3 (histórico) — checklist da Seção 9.**

## 8b. Comandos desta rodada (Tarefa 3.2 — fluxo notebook-only, principal)

Desde a Tarefa 3.2, cada um dos 4 notebooks de FS tem 7 células fixas — a única que você edita normalmente é a **célula 1 (configuração)**:

| # | Célula | O que faz | Edita? |
|---|---|---|---|
| 0 | Imports | `shutil`, `config`, `TimeSeriesFeatureSelector`, etc. | Não |
| 1 | **Configuração** | `model` (Pipeline+strategy), `fs_series_list`, `experiment_id`, `model_name`, `model_parameters` | **Sim — único ponto de edição por rodada** |
| 2 | Sanity-check | Confirma que `Pipeline.get_params(deep=True)` expõe as chaves esperadas | Não |
| 3 | Cópia do ARIMA | `shutil.copy` de `data/result/chamados/<serie>_1arima.pkl` para `data/result/<experiment_id>/` (equivalente ao antigo Passo 0 em bash) | Não |
| 4 | Execução | Laço `GridSearch(...).execution()` por série — mesma lógica de sempre | Não |
| 5 | CSV de métricas | Chama `run_export_metrics_to_csv` (Seção 5) — gera `results/<experiment_id>/metrics.csv`/`_detail.csv` | Não |
| 6 | CSV de features | Chama `run_export_selected_features` (Seção 5) — gera `results/<experiment_id>/selected_features.csv`/`_detail.csv` | Não |

**Passo a passo:**

1. Abra o notebook (ex. `notebook/residual_hydridsystem/arima_mlp_ftest.ipynb`) no VS Code/Jupyter.
2. Edite a célula 1 se precisar trocar séries/`experiment_id`/grid (valores padrão já são `chamados_v4_fs_*` — não reaproveite sem trocar o nome, CLAUDE.md Seção 3.2).
3. "Run All". As células 0-6 rodam em sequência: sanity-check → cópia do ARIMA → grid search → CSV de métricas → CSV de features. Nenhum passo de terminal é necessário além de ativar o `.venv` e abrir o notebook.
4. Repita para os outros 3 notebooks (`arima_mlp_mutual_info.ipynb`, `arima_mlp_rf_embedded.ipynb`, `arima_mlp_lasso.ipynb`).
5. Depois dos 4 rodarem, abra `notebook/compare_fs_results.ipynb` e "Run All" — gera `results/chamados_v4_fs_comparison.csv` comparando o baseline contra as 4 variantes lado a lado (equivalente ao Passo 3 antigo).
6. Checklist da Seção 9 antes de considerar concluído.

**Alternativa via terminal** (mesma função Python por baixo dos dois caminhos — saída idêntica, ver code-review da Tarefa 3.2): os comandos completos continuam documentados nas Seções 3 (cópia do ARIMA), 4 (papermill) e 5 (`export_metrics_to_csv.py`/`export_selected_features.py`/`compare_fs_vs_baseline.py` via CLI). Útil para automação/CI ou para quem prefere não abrir o Jupyter.

## 9. Checklist antes de considerar a rodada concluída

- [ ] Hashes dos `.pkl` dos 5 baselines em `data/result/chamados/` idênticos antes/depois (`sha256sum data/result/chamados/*.pkl`, comparar com um snapshot tirado antes de rodar).
- [ ] `results/baseline_metrics.csv`/`baseline_metrics_detail.csv` da raiz idênticos antes/depois (mesmo hash).
- [ ] `.pkl` novo legível via `metrics.open_fold_result('<experiment_id>')` sem erro, valores batendo com uma leitura direta via `generics.open_saved_result(...)` de pelo menos um arquivo.
- [ ] `pytest` (raiz do projeto, `.venv` ativo) passando integralmente.
- [ ] `experiment_id` usado é novo — nunca reaproveitado (`chamados_v4_fs_*`, não `chamados_v3_fs_*`).
- [ ] Saída de métricas foi para `results/<experiment_id>/`, não sobrescreveu `results/baseline_metrics.csv`.
- [ ] `selected_features.csv`/`selected_features_detail.csv` gerados para as 4 variantes; para `rf_embedded`, conferir se o desvio-padrão de `N_Features_Selected` entre repetições é plausível (variância real esperada, não erro).
- [ ] `results/chamados_v4_fs_comparison.csv` gerado com `--baseline-dir data/result/chamados` (nunca uma pasta `chamados_v*_fs_*`).
- [ ] Resultado de `austres.txt` (lag_size auto = 1) revisado com atenção redobrada antes de tirar conclusões sobre diferença entre métodos de FS para essa série.
