# Arquivo — baseline MLP híbrido (`1amv1`) com `activation='identity'`

## O que é isto

Cópia de arquivamento dos 17 `.pkl` de `1amv1` (baseline ARIMA-MLP híbrido aditivo,
`notebook/residual_hydridsystem/arima_mlp.ipynb`) tal como existiam em
`data/result/chamados/` antes da regeneração da Tarefa 3.9 (2026-07-21).

Esses artefatos foram treinados com `MLPRegressor(activation='identity', solver='lbfgs')`.
`activation='identity'` faz a camada oculta da MLP funcionar como uma combinação linear pura —
**funcionalmente equivalente a uma regressão linear com mais parâmetros, sem capturar
não-linearidade nenhuma no resíduo**.

## Por que foram substituídos

O código-fonte de `arima_mlp.ipynb` foi alterado de `activation='identity'` para
`activation='logistic'` no commit `43dae5076efb2fb8b859ad796ec7d0f143047ece` (2026-07-02),
uma mudança intencional e aprovada (ver CLAUDE.md, nota formal adicionada na Tarefa 3.9). Como
`*.pkl` está no `.gitignore` e o notebook roda com `force=False`, esses 17 arquivos nunca
foram regenerados após a mudança — permaneceram com a configuração antiga (mtime 2026-03-27,
anterior ao commit). Isso foi identificado nas Tarefas 3.8/3.9 como uma divergência entre o
artefato persistido e o código-fonte atual do projeto, e corrigido regenerando os `.pkl` com
`activation='logistic'`.

## Ressalva

`activation='identity'` **não é recomendado como baseline principal** para este projeto — ele
anula a premissa central do sistema híbrido residual (capturar não-linearidade no resíduo do
ARIMA via a MLP). Estes arquivos são preservados aqui só para referência histórica / possível
uso como ablação à parte (ex.: "MLP linear vs. MLP não-linear no resíduo"), caso o orientador
do pesquisador queira essa comparação — não devem ser reintroduzidos em `data/result/chamados/`
nem usados nas comparações oficiais FS-vs-baseline sem essa ressalva sendo declarada
explicitamente no texto.

## Proveniência

- Gerados por: `notebook/residual_hydridsystem/arima_mlp.ipynb`, versão anterior ao commit
  `43dae5076efb2fb8b859ad796ec7d0f143047ece` (2026-07-02).
- Arquivados em: 2026-07-21 (Tarefa 3.9), por cópia byte-a-byte (hash confirmado — ver
  `data/result/chamados_1amv1_pkl_hashes_pre_tarefa_3_9_20260721.txt` para os hashes SHA-256
  dos 85 `.pkl` de `data/result/chamados/` no momento do arquivamento, incluindo estes 17).
