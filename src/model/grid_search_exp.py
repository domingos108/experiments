import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator

import config
from model import generics

from sklearn.exceptions import ConvergenceWarning

# Filter out this specific warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def is_not_sklearn(model):
    # returns True if it is NOT a scikit-learn class/instance
    return not isinstance(model, BaseEstimator)


# Seed base das familias MLP (arima_mlp/*, mlp/*) -- constante do projeto,
# NAO um knob por notebook (CLAUDE.md Secao 3.4). Os notebooks de FS de MLP
# passam `estimator_random_state_base=grid_search_exp.MLP_RANDOM_STATE_BASE`.
# SVR e deterministico e nao usa.
MLP_RANDOM_STATE_BASE = 42


def resolve_lag_size(base_info):
    """
    Resolve o lag_size a usar para uma serie a partir de uma entrada de
    config.BASE_INFORMATION.

    Chave opcional e aditiva: se `fs_lag_size` estiver presente, ela tem
    prioridade (usada pelos experimentos de Feature Selection para expor
    janelas mais profundas de lags aos seletores). Caso contrario, o
    comportamento e identico ao de sempre -- usa `lag_size`, sem alterar
    nenhum dos 5 baselines existentes (ver PLANO_ARQUITETURA.md, Secao 1.3).
    """
    if "fs_lag_size" in base_info:
        return base_info["fs_lag_size"]
    return base_info["lag_size"]


def resolve_lag_size_pct(n_obs, pct=0.10):
    """
    Via PARALELA e ADITIVA a resolve_lag_size(): heuristica de janela por
    percentual fixo do tamanho da serie -- round(pct * n_obs).

    NAO chama nem altera resolve_lag_size()/get_max_lag_to_consider()/
    fs_lag_size -- e um mecanismo independente, exposto aos notebooks via
    o parametro `lag_size_override` de GridSearch (nunca via
    config.BASE_INFORMATION). Motivacao: comparar a janela PACF ('auto')
    contra uma heuristica fixa, isolando exatamente essa variavel.

    `n_obs`: por decisao do pesquisador (2026-08-28), o N da serie COMPLETA
    (len de data/raw). ATENCAO -- isto NAO coincide com a base de calculo de
    get_max_lag_to_consider: aquela funcao (input.py) computa a PACF sobre
    ts_univariate[0:-test_size] (treino+val, exclui teste), entao o 'auto'
    resolve a partir de N-test, nao de N total. A comparacao 'PACF vs.
    percentual' portanto carrega essa diferenca de base (~test_size/N, i.e.
    ~10%); se o pesquisador quiser paridade exata de base, passar
    n_obs = N - test_size no notebook. Registrado como sub-resultado a
    discutir (ver ambiguidade reportada na tarefa).

    Ao contrario de get_max_lag_to_consider, NAO ha teto (min(20, ...)) nem
    guarda de amostra pequena -- e proposital: o caso de uso motivador
    (taylor, lag_size=336) exige exceder o teto de 20. A unica validacao e
    contra entrada degenerada (n_obs<=0, pct fora de (0, 1]); nao ha como
    esta funcao saber o tamanho do treino apos janelamento, entao uma janela
    grande demais para a serie so falha (visivelmente) em create_windowing.

    `pct` e parametrizavel para permitir rodar 0.20 depois sem nova funcao.
    Arredondamento: round() nativo do Python (round-half-to-even).
    """
    if n_obs <= 0:
        raise ValueError(f"n_obs deve ser um inteiro positivo, recebido: {n_obs!r}.")
    if not (0 < pct <= 1):
        raise ValueError(f"pct deve estar em (0, 1], recebido: {pct!r}.")
    return round(pct * n_obs)


class GridSearch:
    def __init__(self,
                 model_class_exp,
                 model,
                 model_parameters,
                 experiment_id,
                 base_name,
                 model_name,
                 force=True,
                 normalize = True,
                 experiment_params = {},
                 model_exec = 10,
                 use_val_slipt_for_prev = False,
                 save_grid_history = True,
                 lag_size_override = None,
                 estimator_random_state_base = None

        ):
        self.model_class_exp = model_class_exp
        self.model = model
        self.model_parameters = model_parameters
        self.experiment_id = experiment_id
        self.base_name = base_name
        self.model_name = model_name
        self.experiment_params = experiment_params
        self.model_exec = model_exec
        self.force = force
        self.normalize = normalize
        self.group_metrics_name = 'val_metrics'
        self.metric = 'RMSE'
        self.use_val_slipt_for_prev = use_val_slipt_for_prev
        self.save_grid_history = save_grid_history
        # Via paralela aditiva (ver resolve_lag_size_pct): quando != None, um
        # int que sobrepoe o lag_size resolvido de config.BASE_INFORMATION,
        # tanto na busca do grid quanto no refit final. None => comportamento
        # byte-a-byte identico ao de sempre (resolve via config).
        self.lag_size_override = lag_size_override
        # Seed opt-in (CLAUDE.md Secao 3.4): quando != None (ex. 42), a
        # i-esima das model_exec repeticoes recebe random_state = base + i no
        # ESTIMADOR (nunca no seletor). None => estocasticidade nao-reprodutivel
        # de sempre. Ver _clone_model_for_rep.
        self.estimator_random_state_base = estimator_random_state_base
        if estimator_random_state_base is not None and is_not_sklearn(model):
            raise ValueError(
                "estimator_random_state_base so se aplica a model sklearn "
                "(Pipeline/estimador nu). model_class nao-sklearn (LSTM/NHITS/"
                "ELM/etc.) usa seu proprio 'random_seed' -- nao passe este parametro."
            )
        self.fold, self.title = generics.format_names(
            experiment_id,
            base_name,
            f'{experiment_params["horizon"]}{model_name}'
        )

    def _clone_model_for_rep(self, params, rep_index):
        """Clona self.model, aplica os hiperparametros da combinacao e --
        quando estimator_random_state_base != None -- injeta um random_state
        deterministico (base + rep_index) no ESTIMADOR: 10 inicializacoes
        distintas mas reproduziveis entre execucoes (CLAUDE.md Secao 3.4).

        Pipeline([selector, estimator]) -> `estimator__random_state` (o step
        DEVE se chamar 'estimator' -- convencao de toda a matriz de FS);
        estimador nu (MLPRegressor via SKlearnModel/Additive) -> `random_state`.

        Limitacoes conhecidas (achados de code-review): (1) so alcanca um nivel
        de aninhamento -- um estimador composto/ensemble no step 'estimator'
        teria seu RNG interno em `estimator__estimator__random_state`, nao
        tocado (nao ocorre na matriz atual, so MLP/SVR simples); (2) SVR nao
        tem `random_state` (determinístico) -> no-op. O seletor NAO e semeado
        de proposito -- rf_embedded/rfecv/mutual_info mantem random_state=None
        (PLANO_ARQUITETURA.md Secao 1.5). Se a seed esta setada mas NENHUMA
        chave de random_state e encontrada num Pipeline, FALHA ALTO (nao
        no-op silencioso) -- sinaliza step mal-nomeado."""
        model_actual = clone(self.model).set_params(**params)
        if self.estimator_random_state_base is not None:
            seed = self.estimator_random_state_base + rep_index
            available = model_actual.get_params()
            if "estimator__random_state" in available:
                model_actual.set_params(estimator__random_state=seed)
            elif "random_state" in available:
                model_actual.set_params(random_state=seed)
            elif hasattr(model_actual, "named_steps"):
                raise ValueError(
                    "estimator_random_state_base setada, mas o Pipeline nao expoe "
                    "'estimator__random_state' -- o step do estimador precisa se "
                    f"chamar 'estimator'. get_params: {sorted(available)}"
                )
            # estimador nu sem random_state (SVR) -> no-op deliberado
        return model_actual

    def _resolve_lag_size(self):
        """lag_size efetivo desta rodada: o override explícito (via paralela
        de resolve_lag_size_pct) quando presente, senão o valor resolvido de
        config.BASE_INFORMATION exatamente como sempre."""
        if self.lag_size_override is not None:
            return self.lag_size_override
        return resolve_lag_size(config.BASE_INFORMATION[self.base_name])

    def _search_params(self):

        experiment_params = self.experiment_params.copy()
        experiment_params['test_size'] = config.TEST_SIZE
        experiment_params['val_size'] = config.VAL_SIZE
        experiment_params['lag_size'] = self._resolve_lag_size()

        target_list_mean_metrics = []
        grid_search_history = []

        model_exec = 1 if self.model_exec < 1 else self.model_exec

        list_params = list(ParameterGrid(self.model_parameters))

        for params in list_params:
            exec_list_metrics = []
            exec_list_all_metrics = []
            # Snapshot ANTES do loop interno: para model_class_exp nao-sklearn,
            # 'params' e o MESMO dict que sera colocado em
            # experiment_params['model_actual_config'] e MUTADO in-place pelo
            # wrapper (ex. neural_forecast_exp.py injeta 'random_seed' e chaves
            # de plumbing) -- capturar depois do loop pegaria a poluicao em vez
            # dos hiperparametros do grid (achado de code-review, Tarefa 3.4).
            params_snapshot = dict(params)

            for rep_index in range(0, model_exec):
                if is_not_sklearn(self.model):
                    experiment_params['model_actual_config'] = params
                    model_actual = self.model
                else:
                    model_actual = self._clone_model_for_rep(params, rep_index)

                model_exp = self.model_class_exp(
                    model_actual,
                    self.experiment_id,
                    self.base_name,
                    self.model_name,
                    self.force,
                    self.normalize,
                    experiment_params
                )

                model_exp.fit_predict()
                metrics_results = model_exp.metrics_results
                val_metrics = metrics_results.get(self.group_metrics_name, {self.metric: np.inf})
                exec_list_metrics.append(val_metrics[self.metric])
                if self.save_grid_history:
                    exec_list_all_metrics.append(dict(val_metrics))

            mean_metric = np.mean(exec_list_metrics)
            target_list_mean_metrics.append(mean_metric)

            if self.save_grid_history:
                with np.errstate(invalid="ignore"):
                    # invalid='ignore': std() sobre uma repeticao com np.inf
                    # (sentinela de metrica ausente, linha acima) produz NaN
                    # corretamente -- so silencia o RuntimeWarning de console,
                    # o NaN persistido continua sinalizando o problema real.
                    val_metric_std = float(np.std(exec_list_metrics))
                grid_search_history.append({
                    'params': params_snapshot,
                    'val_metric_mean': float(mean_metric),
                    'val_metric_std': val_metric_std,
                    'val_metric_reps': [float(v) for v in exec_list_metrics],
                    'val_metrics_reps': exec_list_all_metrics,
                })

        int_arg_min = np.argmin(target_list_mean_metrics)

        return target_list_mean_metrics[int_arg_min], list_params[int_arg_min], grid_search_history

    def execution(self):

        # Idempotencia: mesma regra que grid_seach_multiple_bases ja aplicava,
        # agora tambem no caminho direto GridSearch(...).execution() usado pelos
        # notebooks de FS. Sem isto, force=False era no-op aqui e re-rodar um
        # notebook (ex. ao adicionar 1 serie a fs_series_list) re-executava
        # TODAS as series -- novo sorteio estocastico das ja validadas.
        if not self.force and generics.file_exists(self.title) and os.path.getsize(self.title) > 0:
            print(f"[skip] {self.title} ja existe e force=False -- pulando")
            return

        best_exec_val, best_params, grid_search_history = self._search_params()

        experiment_params = self.experiment_params.copy()
        experiment_params['test_size'] = config.TEST_SIZE
        experiment_params['lag_size'] = self._resolve_lag_size()

        if self.use_val_slipt_for_prev:
            experiment_params['val_size'] = config.VAL_SIZE
        else:
            experiment_params['val_size'] = 0

        predict_results = []
        print(best_params)
        for rep_index in range(0, self.model_exec):

            if is_not_sklearn(self.model):
                experiment_params['model_actual_config'] = best_params
                model_actual = self.model
            else:
                model_actual = self._clone_model_for_rep(best_params, rep_index)

            model_exp_test = self.model_class_exp( 
                model_actual,
                self.experiment_id, 
                self.base_name, 
                self.model_name, 
                self.force,
                self.normalize,
                experiment_params
            )
            model_exp_test.fit_predict()

            entry = {'experiment': model_exp_test, 'val_metric': best_exec_val}
            if self.save_grid_history:
                # Mesma referencia de lista em TODAS as model_exec entradas
                # (nao uma copia por entrada) -- proposital: pickle deduplica
                # objetos repetidos por identidade dentro do mesmo dump, entao
                # isso e ~6-8x mais barato em disco que copiar por entrada
                # (medido em code-review, Tarefa 3.4), ao custo de todas as
                # entradas compartilharem o MESMO objeto. Tratar como
                # somente-leitura -- mutar grid_search_history numa entrada
                # afeta todas as outras.
                entry['grid_search_history'] = grid_search_history
            predict_results.append(entry)

        generics.save_result(self.fold, self.title, predict_results)




def grid_seach_multiple_bases(fit_predict_class, model, normalize, model_parameters,
                              experiment_params,
                              model_exec, model_name, experiment_id,
                              force = True,
                              use_val_slipt_for_prev= True,
                              save_grid_history = True,
                              estimator_random_state_base = None
                              ):

    base_name_list = config.BASE_NAME_LIST
    for base_name in base_name_list:
        print(base_name)

        # O skip por force/file_exists vive agora em GridSearch.execution()
        # (fonte unica) -- este wrapper so precisa construir e chamar.
        exec_gs = GridSearch(
            fit_predict_class,
            model,
            model_parameters,
            experiment_id,
            base_name,
            model_name,
            force,
            normalize,
            experiment_params,
            model_exec = model_exec,
            use_val_slipt_for_prev = use_val_slipt_for_prev,
            save_grid_history = save_grid_history,
            estimator_random_state_base = estimator_random_state_base
        )

        exec_gs.execution()


def load_grid_search_history(pkl_path: Path) -> pd.DataFrame:
    """
    Le o `grid_search_history` persistido por `GridSearch` (Tarefa 3.4) num
    `.pkl` e retorna um DataFrame com uma linha por combinacao testada: uma
    coluna por chave de `params` (ex. `selector__k`), mais
    `val_metric_mean`/`val_metric_std`/`val_metric_reps`/`val_metrics_reps`.
    Colocada aqui (nao em src/utils/) porque le um formato que so
    `GridSearch` produz -- ver PLANO_ARQUITETURA.md Secao 1.6.

    Uso
    ---
        from pathlib import Path
        from model.grid_search_exp import load_grid_search_history

        df = load_grid_search_history(Path('data/result/<experiment_id>/<serie>_<model>.pkl'))
        df.plot(x='selector__k', y='val_metric_mean', yerr='val_metric_std', marker='o')

    Levanta `ValueError` se o `.pkl` nao tiver `grid_search_history` (nao
    presente, ou presente mas vazio) -- ex. gerado com
    `save_grid_history=False`, ou de antes da Tarefa 3.4. Esse historico nao
    e recuperavel retroativamente; re-rode o Grid Search com a instrumentacao
    atual.
    """
    entries = generics.open_saved_result(str(pkl_path))
    entries = entries if isinstance(entries, list) else [entries]

    history = None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("grid_search_history"):
            history = entry["grid_search_history"]
            break

    if not history:
        raise ValueError(
            f"'{pkl_path}' nao tem 'grid_search_history' -- gerado com "
            "save_grid_history=False, ou antes da Tarefa 3.4 (PLANO_ARQUITETURA.md "
            "Secao 1.6). Esse historico nao e recuperavel retroativamente; "
            "re-rode o Grid Search com a instrumentacao atual."
        )

    rows = []
    for combo in history:
        row = dict(combo["params"])
        row["val_metric_mean"] = combo["val_metric_mean"]
        row["val_metric_std"] = combo["val_metric_std"]
        row["val_metric_reps"] = combo["val_metric_reps"]
        row["val_metrics_reps"] = combo.get("val_metrics_reps")
        rows.append(row)

    return pd.DataFrame(rows)