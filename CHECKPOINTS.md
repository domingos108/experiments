# CHECKPOINTS.md — Estado do Projeto (Feature Selection Nativo)

Este documento existe para permitir retomar o trabalho após uma pausa ou perda de sessão de
chat, sem depender do histórico de conversa. Deve ser mantido atualizado a cada tarefa
concluída (ou pausada) — é o complemento "estado atual" ao lado de CLAUDE.md (regras) e
PLANO_ARQUITETURA.md (arquitetura/roadmap).

**Última atualização:** [preencher na próxima sessão, após a auditoria abaixo]
**Branch/estado do Git no momento desta pausa:** `joao_lucas_experiments`, 16 mudanças no
working tree não commitadas (ver Seção 2).

---

## 1. Tabela de Checkpoints

| # | Tarefa | Status | Entregáveis principais |
|---|---|---|---|
| 1 | Scaffold `TimeSeriesFeatureSelector` (f_test, mutual_info) + fix `fs_lag_size` | ✅ Concluída | `src/model/feature_selection.py`, `resolve_lag_size()` em `grid_search_exp.py`, `config.py` |
| 2 | Integração via Pipeline em `arima_mlp_ftest.ipynb` | ✅ Concluída | Primeira execução real: `sunspot`+`airlines`, `f_test`, `experiment_id=chamados_v2_fs_ftest` |
| 2.5 | Triagem de notebooks suspeitos + fix `metrics.py` (bug Windows) | ✅ Concluída | `.gitignore` para `exploratory_analysis/`; `metrics.py` corrigido e testado |
| 2.6 | Reorganização de notebooks + runbook | ✅ Concluída (com 1 lacuna, ver Seção 3) | `arima_mlp.ipynb` restaurado como baseline; `RUNBOOK.md` criado; nota de `diff_kpss` no CLAUDE.md |
| 3 | `rf_embedded`/`lasso` (top-k uniforme) + 4 notebooks dedicados | ✅ Concluída, **nada executado** | `experiment_id` propostos: `chamados_v3_fs_ftest/mutualinfo/rfembedded/lasso` — só preparados |
| 3.1 | Reversão p/ `SelectFromModel` nativo + `fs_lag_size='auto'` + artefato de features + comparação lado a lado | ⚠️ **Status desconhecido — verificar antes de prosseguir** | Prompt enviado; relatório nunca recebido (sessão perdida) |

---

## 2. Estado real do working tree no momento da pausa

Segundo captura de tela do VS Code, 16 arquivos com mudanças não commitadas, incluindo:
`RUNBOOK.md`, `arima_mlp_ftest.ipynb`, `arima_mlp_lasso.ipynb`, `arima_mlp_mutual_info.ipynb`,
`arima_mlp_rf_embedded.ipynb`, `metadata.json` de `chamados_v2_fs_ftest`/`chamados_v3_fs_lasso`/
`chamados_v3_fs_mutualinfo`/`chamados_v3_fs_rfembedded`, `config.py`, `test_mutual_information.py`,
`feature_selection.py`, `test_metrics.py`, `test_feature_selection.py`, `test_grid_search_exp.py`.

**Não há indício visível de artefatos específicos da Tarefa 3.1** (ex. pastas
`chamados_v4_fs_*`) — mas isso não é conclusivo, porque a Tarefa 3.1 poderia ter alterado
arquivos já listados acima (`feature_selection.py`, `config.py`) sem criar pastas novas até
que uma execução real acontecesse. **Não presumir nada — auditar primeiro** (ver Seção 4).

---

## 3. Pendências conhecidas, além do status da Tarefa 3.1

1. **`activation='logistic'` em `arima_mlp.ipynb` nunca foi formalmente documentado no
   CLAUDE.md.** Na Tarefa 2.6, o pré-check encontrou essa mudança não commitada e você decidiu
   preservá-la (em vez de restaurar para `'identity'`) — mas, diferente do `diff_kpss`, essa
   decisão não recebeu uma nota equivalente na Seção 3.4/3.5 do CLAUDE.md. Vale formalizar
   isso na próxima sessão, com o mesmo padrão usado para `diff_kpss` (data, motivo, "não é
   divergência a corrigir").
2. **Item 9 do relatório da Tarefa 2.6** (dois notebooks — `calculate_metrics_v2.ipynb` e
   `arima_exec.ipynb` — voltaram sozinhos ao estado do HEAD) foi explicado por você como
   provável "discard changes" acidental no VS Code. Sem ação pendente, só registrado aqui
   para histórico.
3. Nenhum `.pkl` novo desta fase de Feature Selection foi gerado ainda — toda execução real
   continua pendente, por decisão deliberada de manter a fase manual sob seu controle.

---

## 2. Documentos fundamentais (ordem de leitura recomendada)

1. `CLAUDE.md` — regras imutáveis do repositório (leitura obrigatória, sempre).
2. `PLANO_ARQUITETURA.md` — arquitetura da fase de Feature Selection, arsenal de métodos,
   roadmap original (Seção 3) e convenção de nomenclatura de notebooks (Seção 5).
3. `RUNBOOK.md` — comandos operacionais para rodar/repetir experimentos.
4. Este arquivo (`CHECKPOINTS.md`) — estado atual, sempre a fonte mais recente de "onde
   paramos".

---

## 4. PROMPT DE RETOMADA — colar isto na nova sessão do Claude Code

```markdown
## CONTEXTO OBRIGATÓRIO (ler antes de qualquer ação)
- Leia CLAUDE.md, PLANO_ARQUITETURA.md, RUNBOOK.md e CHECKPOINTS.md, todos na íntegra.
- Este é o retorno de uma sessão anterior que foi perdida antes de eu receber o relatório
  final da Tarefa 3.1. NÃO assuma que a Tarefa 3.1 foi concluída, parcialmente feita, ou não
  iniciada — isso precisa ser verificado, não inferido.

## TAREFA — Auditoria de estado antes de qualquer implementação nova
NÃO implemente nada nesta resposta. Apenas investigue e reporte.

1. Rode `git status`, `git log --oneline -20` e `git diff --stat` para levantar o estado real
   do working tree e do histórico recente.
2. Verifique especificamente, no código atual (não no que eu descrever de memória):
   a. `src/model/feature_selection.py` — `rf_embedded`/`lasso` usam `SelectFromModel`
      (threshold, nº variável de features) ou ainda top-k uniforme (`SelectKBest`-style)?
   b. `config.py` — `fs_lag_size` de `sunspot.txt`/`airlines.txt`/`austres.txt`/
      `coloradoRiver.txt` está como `'auto'` ou com os valores profundos antigos
      (30/20/12/30)?
   c. Existe algum mecanismo/artefato de registro do nº ou índices de features selecionadas
      por execução? Onde?
   d. Existe algum script/notebook de comparação lado a lado (baseline vs. variantes FS)?
   e. Existem pastas `data/result/chamados_v4_fs_*/` ou `results/chamados_v4_fs_*/`, ou
      qualquer `.pkl`/CSV gerado por uma execução real desta fase de Feature Selection?
   f. O `.gitignore`, os testes (`pytest`) e os hashes dos 5 baselines em `data/result/
      chamados/` seguem íntegros?
3. Verifique também a pendência registrada em CHECKPOINTS.md Seção 3, item 1: confirme se
   `arima_mlp.ipynb` ainda usa `activation='logistic'` e se há alguma nota sobre isso no
   CLAUDE.md (esperado: NÃO há, essa é uma lacuna conhecida a resolver).

## FORMATO DE RESPOSTA ESPERADO
1. Diagnóstico claro e objetivo: para cada item (2a-2f, 3), indicar se está "implementado
   conforme a Tarefa 3.1", "ainda no estado da Tarefa 3 (não revertido)", ou "inexistente".
2. Nenhuma ação de escrita nesta resposta — nem commit, nem edição de código, nem execução de
   notebook.
3. Ao final, proponha um plano objetivo de próximos passos com base no diagnóstico (ex.: "a
   Tarefa 3.1 não foi iniciada, retomar do zero" ou "a Tarefa 3.1 foi parcialmente feita,
   faltando X e Y") — e aguarde minha confirmação antes de agir.
```

---

## 5. Depois da auditoria

Assim que o diagnóstico voltar, atualizar a Tabela de Checkpoints (Seção 1) deste arquivo com
o status real da Tarefa 3.1, e então decidir com base nisso:
- Se a Tarefa 3.1 não foi feita → reenviar o prompt original da Tarefa 3.1 (mantido no
  histórico de `PROMPT_PATTERN_claude_code.md`, se você tiver esse arquivo salvo).
- Se foi parcialmente feita → montar uma Tarefa 3.2 cobrindo só o que falta, seguindo o mesmo
  padrão de prompt já estabelecido (pré-checks, escopo negativo explícito, formato de
  resposta).
- Se foi totalmente feita → seguir para a primeira execução manual real via `RUNBOOK.md`.

Não deixar de **atualizar este arquivo e commitar** ao final de cada tarefa daqui em diante —
é o que teria evitado a incerteza desta pausa.