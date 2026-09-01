# Arquivo — baselines de `taylor.txt` gerados com config incorreto (`MS` / `m=12` / `lag_size='auto'`)

## O que é isto

Cópia de arquivamento dos 5 `.pkl` de baseline de `taylor.txt` tal como existiam em
`data/result/chamados/` antes da correção da entrada de `taylor.txt` em
`src/config.py` (`BASE_INFORMATION`), em 2026-08-28:

| `.pkl` | Arquitetura | mtime original |
|---|---|---|
| `taylor_1arima.pkl` | ARIMA | 2026-03-27 |
| `taylor_1mlp.pkl` | MLP single | 2026-07-21 |
| `taylor_1svr.pkl` | SVR single | 2026-03-27 |
| `taylor_1amv1.pkl` | ARIMA-MLP híbrido aditivo | 2026-07-21 |
| `taylor_1as.pkl` | ARIMA-SVR híbrido aditivo | 2026-03-27 |

Hashes SHA-256 byte-a-byte em `../chamados_taylor_pkl_hashes_pre_config_fix_20260828.txt`.

## Por que foram arquivados

`taylor.txt` é a série de demanda elétrica **semi-horária** de Taylor (2003) — 48 observações
por dia, 4032 no total. A entrada em `BASE_INFORMATION` estava, até 2026-08-28:

```python
'taylor.txt': {"freq": "MS", 'm': 12, 'lag_size': 'auto'},   # provisório
```

`freq='MS'` (mensal) e `m=12` eram **placeholders incorretos** para dado semi-horário — nunca
foram revistos. Consequências nos artefatos acima:

- `taylor_1arima.pkl` / `taylor_1as.pkl` / `taylor_1amv1.pkl`: o `pm.auto_arima` rodou com
  `m=12` (período sazonal de 12) sobre uma série cujos ciclos reais são 48 (diário) e 336
  (semanal) — modelo linear estruturalmente mal-especificado.
- Todos os 5: `lag_size='auto'` resolve, via `get_max_lag_to_consider`, para no máximo 20 lags
  (teto de `safe_nlags`) — ~10 horas de histórico, cega para os ciclos de 48 e 336.

A entrada foi corrigida (decisão do pesquisador, 2026-08-28) para:

```python
'taylor.txt': {"freq": "30min", 'm': 48, 'lag_size': 336},
```

`freq='30min'` (granularidade real), `m=48` (ciclo intra-diário — o ciclo semanal de 336 não
entra no ARIMA porque `auto_arima` com `m>=336` é inviável; fica coberto pelo `lag_size`
profundo), `lag_size=336` **fixo** (não `'auto'`, para exceder deliberadamente o teto de 20).

## Ressalva

Estes 5 `.pkl` **não devem ser reintroduzidos em `data/result/chamados/`** nem usados em
nenhuma comparação (FS-vs-baseline ou benchmark) — os números são de um modelo mal-especificado
para a natureza da série. Ficam aqui só como registro histórico. `taylor` está fora da primeira
leva de execução `'auto'` (é estruturalmente mais pesada: N=4032, janela de 336 colunas) e terá
uma rodada dedicada, precedida do pré-check de custo real de `rfecv` com `lag_size=336`.

## Impacto na referência de integridade

As 5 entradas `taylor_*.pkl` foram **removidas** de
`../chamados_baseline_reference_hashes.json` nesta mesma data — não são mais um baseline
protegido (a série sai da referência até ser regenerada sob o config correto). Ver a nota
atualizada no campo `note` daquele arquivo e CLAUDE.md Seção 3.8.

## Proveniência

- Gerados por: `notebook/single_models/arima_exec.ipynb`, `mlp_exec.ipynb`, `svr_exec.ipynb`,
  `notebook/residual_hydridsystem/arima_mlp.ipynb`, `arima_svr.ipynb`, com `taylor.txt` na
  `BASE_NAME_LIST` (rodada ampla de 17 séries, hoje comentada).
- Arquivados em: 2026-08-28, por `mv` (byte-a-byte — hashes confirmados no `.txt` acima).
