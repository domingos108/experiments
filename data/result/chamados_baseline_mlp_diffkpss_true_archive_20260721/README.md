# Arquivo — baseline MLP single (`1mlp`) com `diff_kpss=True`

## O que é isto

Cópia de arquivamento dos 17 `.pkl` de `1mlp` (baseline MLP single, `notebook/single_models/mlp_exec.ipynb`)
tal como existiam em `data/result/chamados/` antes da regeneração da Tarefa 5.1 (2026-07-21).

Esses artefatos foram treinados com `experiment_params['diff_kpss'] = True`.

## Por que foram substituídos

O código-fonte de `mlp_exec.ipynb` foi alterado de `diff_kpss=True` para `diff_kpss=False`
no commit `43dae5076efb2fb8b859ad796ec7d0f143047ece` (2026-07-02) — a mesma mudança, no mesmo
commit, que corrigiu `activation` em `arima_mlp.ipynb` (Tarefa 3.9). `diff_kpss=False` é uma
mudança intencional e aprovada, por solicitação do orientador do pesquisador (`diff_kpss=True`
causava problemas em alguns casos) — já documentada como comportamento correto no CLAUDE.md
Seção 3.5 desde então. Como `*.pkl` está no `.gitignore` e o notebook roda com `force=False`,
estes 17 arquivos nunca foram regenerados após a mudança — permaneceram com a configuração
antiga (mtime 2026-03-27, anterior ao commit). Identificado como pendência na Tarefa 3.9
(pré-check item 3) e corrigido aqui, seguindo o mesmo processo já aplicado ao baseline
`1amv1` (Tarefa 3.9).

## Ressalva

Estes arquivos não devem ser reintroduzidos em `data/result/chamados/` nem usados nas
comparações oficiais "MLP com FS vs. MLP baseline" (família MLP single, Tarefa 5) sem essa
ressalva sendo declarada explicitamente — eles refletem uma configuração diferente
(`diff_kpss=True`) da que o restante do projeto usa hoje (`diff_kpss=False`, CLAUDE.md Seção
3.5), então não são comparáveis diretamente aos `.pkl` de FS do MLP single já preparados na
Tarefa 5, que usam `diff_kpss=False`.

## Proveniência

- Gerados por: `notebook/single_models/mlp_exec.ipynb`, versão anterior ao commit
  `43dae5076efb2fb8b859ad796ec7d0f143047ece` (2026-07-02).
- Arquivados em: 2026-07-21 (Tarefa 5.1), por cópia byte-a-byte (hash confirmado — ver
  `data/result/chamados_1mlp_pkl_hashes_pre_tarefa_5_1_20260721.txt` para os hashes SHA-256
  dos 85 `.pkl` de `data/result/chamados/` no momento do arquivamento, incluindo estes 17).
