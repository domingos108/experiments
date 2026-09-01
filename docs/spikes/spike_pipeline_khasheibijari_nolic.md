# Spike — `Pipeline([selector, estimador])` em `KhasheiBijariHybrid`/`NoLiCHybrid`

**Tarefa 15/16 do roadmap.** Pergunta: a mesma composição `Pipeline([TimeSeriesFeatureSelector, estimador])`
que já funciona sem mudança em `Additive`/`SKlearnModel` (ARIMA-MLP, MLP, SVR, ARIMA-SVR — Tarefas 4-8)
funciona também nos 2 wrappers híbridos não-aditivos (`KhasheiBijariHybrid`, `NoLiCHybrid`,
`src/model/hybrid_system_exp.py`), que constroem matrizes de entrada combinadas fora do fluxo padrão?

**Resposta curta:** `KhasheiBijariHybrid` sim, sem nenhuma mudança. `NoLiCHybrid` não — quebra no
estágio combinador (M_c), com uma causa raiz precisa e um caminho de correção nativo claro.

---

## Parte A.1 — `KhasheiBijariHybrid`: estrutura real

`format_khashei_bijari_input()` (`hybrid_system_exp.py:618-697`) monta a matriz combinada
`[residual_lag_1..n1, linear_forecast_t, series_lag_1..m1]` **inteiramente em `pandas`, antes** de
qualquer chamada ao modelo. `fit_predict()` (linha 745) então delega para:

```python
# hybrid_system_exp.py:790-801
(
    train_predict, val_predict, test_predict, time_exec
) = generics.fit_predict_ml_schemma(
    self.model, x_train, y_train, x_val, x_test
)
```

`generics.fit_predict_ml_schemma` (`generics.py:16-35`) só chama `model.fit(x_train, y_train)` /
`model.predict(...)` — a mesma função 100% agnóstica já usada por `Additive`, `SKlearnModel` e
`NonLinear`. Não há nenhum acesso a atributo interno de `MLPRegressor`/`SVR` (`.coefs_`,
`.support_vectors_`, `.n_layers_` etc.) em nenhum lugar da classe.

**Teste real** (`tests/model/test_hybrid_system_exp.py::TestKhasheiBijariHybridAcceptsPipeline`,
dado real de `airlines`, `.pkl` do ARIMA copiado para um `MODEL_DATA_PATH` temporário):

```python
model = Pipeline([
    ("selector", TimeSeriesFeatureSelector(strategy="f_test", k=3)),
    ("estimator", MLPRegressor(hidden_layer_sizes=(5,), max_iter=200, random_state=42)),
])
# ... GridSearch(hybrid_system_exp.KhasheiBijariHybrid, model, ...).execution()
```

Resultado: `n_features_in_ == 41` (`n1_lags=20 + linear_forecast_t=1 + m1_lags=20`, default
`lag_size_formated` de airlines no contexto híbrido) e `selected_indices_.shape[0] == 3` —
confirma que o seletor recebeu a **matriz combinada inteira**, não uma fatia pré-filtrada, e
`test_metrics` não veio vazio (resultado calculado de verdade).

**Achado extra (não previsto no plano original, encontrado rodando o teste):**
`selected_indices_` retornou `[20, 29, 40]` — o índice **20 é a coluna `linear_forecast_t`**, não
um lag de resíduo ou de série. Ou seja, `f_test` considerou a previsão linear do ARIMA uma das 3
"melhores" features nessa execução real, ao lado de 2 lags de série. Isso é matematicamente
legítimo (o seletor não distingue semântica de coluna, só poder preditivo), mas é uma decisão de
escopo que ainda não foi tomada explicitamente: **o FS deveria poder "descartar" a previsão linear
do ARIMA junto com os lags, ou essa coluna deveria ser protegida (sempre mantida)?** Reportado como
ambiguidade — ver seção final.

**Conclusão A.1:** `KhasheiBijariHybrid` **encaixa sem nenhuma mudança de código**. Próxima tarefa
pode seguir o mesmo roteiro já usado nas Tarefas 5-7 (notebooks `khashei_bijari_ftest.ipynb` etc.),
com a ressalva do achado acima a decidir antes.

---

## Parte A.2 — `NoLiCHybrid`: estrutura real e causa raiz da quebra

`NoLiCHybrid.fit_predict()` tem 3 estágios (`hybrid_system_exp.py:1052-1228`):

1. **Estágio 1** — carrega ARIMA pré-treinado (`input_linear_info`, mesmo de sempre).
2. **Estágio 2 (M_NL)** — treina `self.model` sobre os resíduos via `generics.fit_predict_model`
   (linha 1099) — **mesmo padrão agnóstico** que já funciona em `Additive`. Isolado, este estágio
   funcionaria com `Pipeline` sem problema.
3. **Estágio 3 (M_c, combinador)** — **aqui está a quebra.** `_grid_search_mlp()`
   (`hybrid_system_exp.py:922-976`) monta um `GridSearchCV` **interno e hardcoded**:

```python
# hybrid_system_exp.py:956-968
param_grid = {
    'hidden_layer_sizes': [(i,) for i in range(1, 31)],
    'solver': ['lbfgs', 'adam']
}
grid_search = GridSearchCV(
    estimator=clone(self.model),   # <- clone do MESMO self.model do estágio 2
    param_grid=param_grid,
    cv=ps,
    ...
)
grid_search.fit(X_combined, y_combined)   # linha 970 -- é aqui que explode
```

`param_grid` usa os nomes de parâmetro do **próprio `MLPRegressor`**, sem prefixo `estimator__` —
assume implicitamente que `self.model` É um `MLPRegressor` bruto. Quando `self.model` é um
`Pipeline`, `hidden_layer_sizes`/`solver` não existem no nível raiz de `Pipeline.get_params()` (só
`selector__*`/`estimator__*` existem), então `Pipeline.set_params(hidden_layer_sizes=...)` levanta
erro dentro do próprio `GridSearchCV.fit()`.

**Teste real** (`tests/model/test_hybrid_system_exp.py::TestNoLiCHybridBreaksWithPipeline`, mesmo
setup de dado real):

```
ValueError: Invalid parameter 'hidden_layer_sizes' for estimator Pipeline(steps=[
    ('selector', TimeSeriesFeatureSelector(k=3)),
    ('estimator', MLPRegressor(hidden_layer_sizes=(5,), random_state=42))
]). Valid parameters are: ['memory', 'steps', 'transform_input', 'verbose'].
```

Traceback real: `fit_predict` (linha 1179, chamada a `self._grid_search_mlp`) →
`_grid_search_mlp` (linha 970, `grid_search.fit(...)`) → sklearn `Pipeline.set_params` →
`ValueError`. **Falha alto e claro (não silenciosamente)** — não é o pior caso (resultado errado
sem erro), é uma falha visível logo na primeira tentativa de grid search do combinador, antes de
qualquer `.pkl` ser gravado.

**Conclusão A.2:** `NoLiCHybrid` **não encaixa** — precisa de trabalho nativo em
`_grid_search_mlp` antes de qualquer notebook de FS ser criado para essa família.

---

## Parte A.3 — Desenho da correção nativa (NÃO implementado — só proposta)

Duas perguntas distintas, que não devem ser resolvidas juntas sem confirmação:

### 1. Correção mecânica (necessária de qualquer forma)

Tornar `param_grid` ciente de composição, sem monkey patch — método explícito, novo, na própria
classe (conforme CLAUDE.md Seção 4.1):

```python
def _build_param_grid_for_model(self, base_param_grid):
    """Prefixa as chaves do grid com o nome do ultimo step quando self.model
    for um Pipeline -- convencao nativa do sklearn (Pipeline.get_params(deep=True)/
    clone().set_params() ja resolvem 'stepname__param' automaticamente, mesma
    convencao ja usada pelos notebooks de FS via GridSearch externo)."""
    if isinstance(self.model, Pipeline):
        final_step_name = self.model.steps[-1][0]
        return {f"{final_step_name}__{k}": v for k, v in base_param_grid.items()}
    return base_param_grid
```

E em `_grid_search_mlp`, trocar o `param_grid` hardcoded por
`self._build_param_grid_for_model({'hidden_layer_sizes': [...], 'solver': [...]})`.

### 2. Decisão de escopo em aberto (precisa do pesquisador, não é só código)

`self.model` é reaproveitado **duas vezes** com propósitos diferentes: estágio 2 (resíduos —
onde o roadmap do CLAUDE.md Seção 4 diz que o FS deve atuar) e estágio 3 (combinador — cuja
matriz de entrada é `[lags de L̂, lags de Ŷ^N]`, **não** lags de resíduo). Se `self.model` continuar
sendo clonado com o seletor incluído para o combinador:
- o seletor do combinador vai escolher entre colunas de previsão linear/não-linear, um espaço de
  features conceitualmente diferente do que o roadmap descreve;
- a correção mecânica do item 1 sozinha bastaria para não quebrar, mas o resultado seria
  "roda sem erro, mas seleciona features num contexto para o qual o seletor não foi pensado" —
  exatamente o risco de "resultado silenciosamente incorreto" que esta tarefa pediu para investigar.

**Duas opções, sem recomendação unilateral meta (ver ambiguidade reportada abaixo):**
- **Opção A** — manter `self.model` (com seletor) também no combinador; aceitar que FS passa a
  atuar nas duas matrizes.
- **Opção B** — escopar FS só ao estágio residual: para o combinador, `_grid_search_mlp` sempre
  clona só a parte `estimator` (`self.model.named_steps['estimator'] if isinstance(self.model, Pipeline) else self.model`),
  nunca o seletor.

**Nota informativa (fora do escopo de correção agora):** `_grid_search_mlp` já é, mesmo sem FS, um
grid search **paralelo e independente** do `GridSearch`/`model_parameters` padrão usado pelas
outras 4 famílias — não há hoje nenhuma forma do notebook configurar a grade de 30 topologias ×
2 solvers do combinador via `model_parameters` externo. Isso não é um problema introduzido por
esta investigação, só um lembrete de que `NoLiCHybrid` já era estruturalmente diferente das outras
famílias antes de FS entrar em cena.

---

## Recomendação de próximo passo

1. **`KhasheiBijariHybrid`**: pode seguir diretamente o roteiro das Tarefas 5-7 (criar
   `khashei_bijari_ftest.ipynb`/`_mutual_info.ipynb`/etc.) — nenhum redesenho necessário. Decidir
   antes, com o pesquisador, a ambiguidade do achado A.1 (proteger `linear_forecast_t` do FS ou não).
2. **`NoLiCHybrid`**: implementar a correção mecânica (item A.3.1) **e** decidir a Opção A vs B
   (item A.3.2) como uma tarefa de implementação própria, antes de qualquer notebook de FS para
   essa família — não pode seguir o roteiro padrão ainda.

## Testes: formalizados como regressão permanente

Os 2 testes do spike têm valor além desta investigação — travam a suposição estrutural que outra
mudança futura poderia quebrar silenciosamente. Formalizados em
`tests/model/test_hybrid_system_exp.py`:
- `TestKhasheiBijariHybridAcceptsPipeline` (espera sucesso, com asserts sobre `n_features_in_`/
  `selected_indices_`, não só ausência de erro).
- `TestNoLiCHybridBreaksWithPipeline` (documenta o comportamento ATUAL, quebrado, via
  `pytest.raises` com a mensagem exata — se a correção nativa acima for implementada, este teste
  precisa ser atualizado para esperar sucesso, não removido silenciosamente).

## Ambiguidades — reportadas, não resolvidas

1. **`linear_forecast_t` sujeito a FS em `KhasheiBijariHybrid`** (achado A.1): o seletor pode
   descartar a própria previsão do ARIMA se o critério estatístico não a favorecer. Não decidido
   se isso é aceitável ou se essa coluna deveria ser protegida.
2. **Opção A vs B em `NoLiCHybrid`** (item A.3.2): se o FS deve ou não se propagar para o estágio
   combinador. Ambas as opções são tecnicamente viáveis e sem monkey patch; a escolha é de design
   experimental, não de engenharia.
