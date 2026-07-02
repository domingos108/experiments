# RUNBOOK.md — Rodando Experimentos de Feature Selection

Guia prático para você (pesquisador) repetir/variar experimentos de Feature Selection sem reconstruir o raciocínio do zero. Para as regras que **não** podem mudar, veja [CLAUDE.md](CLAUDE.md); para o desenho da arquitetura, [PLANO_ARQUITETURA.md](PLANO_ARQUITETURA.md).

---

## 1. Trocar a série

Hoje restrito a `FS_DEV_SERIES` (`airlines`, `austres`, `coloradoRiver`, `sunspot` — definido em [tests/model/conftest.py](tests/model/conftest.py)). Expandir essa lista é decisão explícita futura, não algo a fazer implicitamente ao seguir este runbook.

No notebook (ex. `arima_mlp_ftest.ipynb`), a série entra na célula de execução:

```python
fs_series_list = ['sunspot.txt', 'airlines.txt']   # troque aqui
```

Se a série usa `fs_lag_size` (janela profunda de lags, ver Seção 1.3 do PLANO_ARQUITETURA.md), confirme que ela está registrada em `src/config.py → BASE_INFORMATION['<serie>.txt']['fs_lag_size']`. Hoje só `sunspot.txt` (30) e `airlines.txt` (20) têm essa chave. Antes de rodar uma 3ª série pela primeira vez, calcule quantas linhas de treino sobram após o janelamento duplo (ver Tarefa 2: script rápido usando `input.open_format_train_val_test` — não assuma que "cabe", meça).

## 2. Trocar a estratégia de FS e os hiperparâmetros do seletor

Estratégia (fixa por notebook — nunca misturar no mesmo grid):

```python
model = Pipeline([
    ('selector', TimeSeriesFeatureSelector(strategy='mutual_info')),  # 'f_test' | 'mutual_info' hoje
    ('estimator', MLPRegressor(activation='logistic', solver='lbfgs')),
])
```

Hiperparâmetros via `model_parameters`, convenção `selector__*`/`estimator__*`:

```python
model_parameters = {
    'selector__k': [5, 10, 15],
    'estimator__hidden_layer_sizes': [10, 20, 50],
    'estimator__max_iter': [1000],
}
```

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

Se o híbrido depende de ARIMA pré-treinado (`Additive`, via `experiment_params['linear_model_name']`), o `.pkl` do ARIMA precisa existir **sob o mesmo `experiment_id` novo**. Como o ARIMA em si não muda com `fs_lag_size`/estratégia de FS, copie (não retreine) do experimento que já tem esse ARIMA:

```bash
mkdir -p data/result/<novo_experiment_id>
cp data/result/chamados/<serie>_1arima.pkl data/result/<novo_experiment_id>/<serie>_1arima.pkl
```

## 4. Rodar o notebook via papermill

Comando real, confirmado na Tarefa 2 (roda a partir da raiz do projeto, `.venv` ativo):

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

## 5. Agregar métricas e exportar CSV

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

`PYTHONIOENCODING=utf-8` evita um erro de encoding do console Windows ao imprimir `✓` — não é um problema do script, só do terminal.

**Nunca** rode `export_metrics_to_csv.py` sem `--result-dir`/`--output` explícitos para um experimento novo — o padrão sem esses parâmetros escreve em `results/baseline_metrics.csv`, sobrescrevendo o relatório dos 5 baselines originais.

## 6. Convenção de nomenclatura de notebooks (FS × híbrido)

Detalhada na Seção 5 do [PLANO_ARQUITETURA.md](PLANO_ARQUITETURA.md#5-convenção-de-nomenclatura-de-notebooks-por-combinação-fs--híbrido). Resumo:

```
<hibrido>_<estrategia_fs>.ipynb
```

| Notebook | Status |
|---|---|
| `arima_mlp_ftest.ipynb` | Implementado |
| `arima_mlp_mutual_info.ipynb` | Não implementado — replicar `arima_mlp_ftest.ipynb` trocando `strategy` |
| `arima_mlp_rf_embedded.ipynb` | Não implementado — depende da estratégia `rf_embedded` ainda não existir em `feature_selection.py` (Tarefa 3) |
| `arima_mlp_lasso.ipynb` | Não implementado — idem, estratégia `lasso` (Tarefa 3) |
| `arima_mlp_rfecv.ipynb` | Não implementado — idem, estratégia `rfecv` (Tarefa 4) |
| `arima_svr_ftest.ipynb` | Não implementado |
| `nonlinear_mlp_ftest.ipynb` (e variantes) | Não implementado — só quando o roadmap de combinações não-lineares de 2–3 estágios (CLAUDE.md Seção 4) for implementado |

`arima_mlp.ipynb`/`arima_svr.ipynb` (sem sufixo) **nunca** recebem essas mudanças — são os baselines intocáveis da Seção 3 do CLAUDE.md.

## 7. Checklist antes de considerar a rodada concluída

- [ ] Hashes dos `.pkl` dos 5 baselines em `data/result/chamados/` idênticos antes/depois (`sha256sum data/result/chamados/*.pkl`, comparar com um snapshot tirado antes de rodar).
- [ ] `results/baseline_metrics.csv`/`baseline_metrics_detail.csv` da raiz idênticos antes/depois (mesmo hash).
- [ ] `.pkl` novo legível via `metrics.open_fold_result('<experiment_id>')` sem erro, valores batendo com uma leitura direta via `generics.open_saved_result(...)` de pelo menos um arquivo.
- [ ] `pytest` (raiz do projeto, `.venv` ativo) passando integralmente.
- [ ] `experiment_id` usado é novo — nunca reaproveitado.
- [ ] Saída de métricas foi para `results/<experiment_id>/`, não sobrescreveu `results/baseline_metrics.csv`.
