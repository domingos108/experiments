"""
Via PARALELA e ADITIVA de definicao de janela de lags -- heuristica de
percentual fixo (round(pct * N)) como alternativa ao mecanismo
'auto'/PACF (get_max_lag_to_consider) e a janela fixa manual de
BASE_INFORMATION.

Duas pecas, ambas estritamente aditivas (PLANO_ARQUITETURA.md Secao 1.3
e o precedente de fs_lag_size):

1. `resolve_lag_size_pct(n_obs, pct=0.10)` -- funcao IRMA de
   `resolve_lag_size()`, nunca a chama nem altera. Devolve round(pct *
   n_obs). Decisao do pesquisador (2026-09-02): os notebooks pct10
   passam `n_obs = N - test_size` (== N - int(config.TEST_SIZE * N)), a
   MESMA base que get_max_lag_to_consider usa (PACF sobre
   ts_univariate[0:-test_size]) -- isola exatamente "PACF vs. percentual".

2. `GridSearch(..., lag_size_override=None)` -- parametro novo no fim da
   assinatura. Quando None (default), o comportamento e byte-a-byte
   identico ao de hoje (resolve via config.BASE_INFORMATION). Quando um
   int, sobrepoe o lag_size resolvido do config TANTO na fase de busca
   do grid (_search_params) QUANTO no refit final (execution).

Regra nao-negociavel: `resolve_lag_size()` original -- zero mudanca de
comportamento (coberto tambem por TestResolveLagSizeRegression em
test_grid_search_exp.py; aqui travamos o WIRING do override).
"""

import pytest

import config
from model import generics, grid_search_exp
from model.grid_search_exp import resolve_lag_size, resolve_lag_size_pct


class TestResolveLagSizePct:
    @pytest.mark.parametrize(
        "n_obs, expected",
        [
            # n_obs = N - test_size (N - int(0.1*N)) -- a base que os notebooks
            # pct10 passam, paridade com get_max_lag_to_consider.
            (130, 13),    # airlines       (N=144, test=14)
            (81, 8),      # austres        (N=89,  test=8)
            (670, 67),    # coloradoRiver  (N=744, test=74)
            (260, 26),    # sunspot        (N=288, test=28)
            (129, 13),    # windspeedfortaleza (N=143, test=14)
            (1070, 107),  # samurec        (N=1188, test=118)
        ],
    )
    def test_returns_ten_percent_of_train_n_rounded(self, n_obs, expected):
        assert resolve_lag_size_pct(n_obs) == expected

    def test_pct_is_parametrizable_for_later_20pct_run(self):
        assert resolve_lag_size_pct(4032, pct=0.20) == 806
        assert resolve_lag_size_pct(144, pct=0.20) == 29  # round(28.8)

    def test_uses_python_round_bankers_rounding_at_exact_half(self):
        """Documenta o comportamento de arredondamento: round() nativo do
        Python (round-half-to-even). Nenhuma das 7 series do estudo cai num
        .5 exato, mas o contrato fica travado e visivel."""
        assert resolve_lag_size_pct(25, pct=0.10) == 2   # round(2.5) -> 2
        assert resolve_lag_size_pct(35, pct=0.10) == 4   # round(3.5) -> 4

    def test_returns_plain_int(self):
        result = resolve_lag_size_pct(1188)
        assert isinstance(result, int)

    @pytest.mark.parametrize("bad_n", [0, -1, -100])
    def test_rejects_non_positive_n_obs(self, bad_n):
        """Guarda de entrada degenerada (achado de code-review): n_obs <= 0
        nunca e uma serie valida -- falha alto e claro em vez de devolver 0/
        negativo que so quebraria la adiante em create_windowing."""
        with pytest.raises(ValueError):
            resolve_lag_size_pct(bad_n)

    @pytest.mark.parametrize("bad_pct", [0, -0.1, 1.0001, 2])
    def test_rejects_pct_outside_open_zero_to_one(self, bad_pct):
        """pct fora de (0, 1]: 0 => janela vazia; > 1 => janela maior que a
        propria serie. Ambos sao erro de chamada, nao um caso de uso."""
        with pytest.raises(ValueError):
            resolve_lag_size_pct(1000, pct=bad_pct)

    def test_pct_of_exactly_one_is_allowed(self):
        """pct=1.0 (janela = serie inteira) e o limite superior legitimo --
        degenerado mas nao um erro de chamada."""
        assert resolve_lag_size_pct(100, pct=1.0) == 100


class _LagSizeCapturingModelExp:
    """Test double de model_class_exp: registra o experiment_params['lag_size']
    que o GridSearch entregou, em CADA chamada (fase de busca do grid E refit
    final). Usa o proprio parametro do grid como RMSE de validacao (double
    deterministico, sem treino real)."""

    captured = []

    def __init__(self, model, experiment_id, base_name, model_name, force, normalize, experiment_params):
        self.model = model
        type(self).captured.append(experiment_params.get("lag_size"))
        self.metrics_results = None

    def fit_predict(self):
        self.metrics_results = {
            "val_metrics": {"RMSE": float(self.model.constant)},
            "test_metrics": {"RMSE": 1.0},
        }


@pytest.fixture
def capturing_model_exp():
    _LagSizeCapturingModelExp.captured = []
    return _LagSizeCapturingModelExp


def _make_grid_search(tmp_path, monkeypatch, model_exp, lag_size_override=None):
    from sklearn.dummy import DummyRegressor

    monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
    return grid_search_exp.GridSearch(
        model_exp,
        DummyRegressor(strategy="constant", constant=0.0),
        {"constant": [1.0]},
        "fake_experiment_lag_override",
        "airlines.txt",
        "testmodel",
        force=True,
        normalize=True,
        experiment_params={"horizon": 1, "diff_kpss": False},
        model_exec=1,
        lag_size_override=lag_size_override,
    )


class TestLagSizeOverrideWiring:
    def test_defaults_to_none(self, tmp_path, monkeypatch, capturing_model_exp):
        exec_gs = _make_grid_search(tmp_path, monkeypatch, capturing_model_exp)
        assert exec_gs.lag_size_override is None

    def test_none_resolves_via_config_exactly_as_before(self, tmp_path, monkeypatch, capturing_model_exp):
        """Sem override: o lag_size entregue e o mesmo que resolve_lag_size
        devolve para a serie no config -- comportamento inalterado."""
        expected = resolve_lag_size(config.BASE_INFORMATION["airlines.txt"])

        exec_gs = _make_grid_search(tmp_path, monkeypatch, capturing_model_exp)
        exec_gs.execution()

        assert capturing_model_exp.captured  # houve chamadas
        assert set(capturing_model_exp.captured) == {expected}

    def test_int_override_replaces_config_value_in_all_phases(self, tmp_path, monkeypatch, capturing_model_exp):
        """Com override=7: TODA chamada (busca do grid + refit final) recebe 7,
        nunca o 'auto' do config."""
        exec_gs = _make_grid_search(tmp_path, monkeypatch, capturing_model_exp, lag_size_override=7)
        exec_gs.execution()

        assert capturing_model_exp.captured
        assert set(capturing_model_exp.captured) == {7}

    def test_none_still_calls_resolve_lag_size(self, tmp_path, monkeypatch, capturing_model_exp):
        calls = []
        real = grid_search_exp.resolve_lag_size
        monkeypatch.setattr(
            grid_search_exp, "resolve_lag_size",
            lambda base_info: calls.append(base_info) or real(base_info),
        )
        _make_grid_search(tmp_path, monkeypatch, capturing_model_exp).execution()
        assert calls  # sem override, o caminho antigo e exercitado

    def test_int_override_never_calls_resolve_lag_size(self, tmp_path, monkeypatch, capturing_model_exp):
        calls = []
        real = grid_search_exp.resolve_lag_size
        monkeypatch.setattr(
            grid_search_exp, "resolve_lag_size",
            lambda base_info: calls.append(base_info) or real(base_info),
        )
        _make_grid_search(
            tmp_path, monkeypatch, capturing_model_exp, lag_size_override=7
        ).execution()
        assert calls == []  # override curto-circuita o caminho do config


class TestLagSizeOverrideRealPipeline:
    """End-to-end com o pipeline real (SKlearnModel + LinearRegression): o
    override precisa produzir um df_train com o numero de colunas de lag
    pedido, atraves do fluxo de producao de verdade."""

    def test_override_controls_real_feature_matrix_width(self, tmp_path, monkeypatch):
        from sklearn.linear_model import LinearRegression
        from model import single_ml_model_exp

        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        exec_gs = grid_search_exp.GridSearch(
            single_ml_model_exp.SKlearnModel,
            LinearRegression(),
            {"fit_intercept": [True]},
            "fake_experiment_lag_override_real",
            "airlines.txt",
            "testmodel",
            force=True,
            normalize=True,
            experiment_params={"horizon": 1, "diff_kpss": False},
            model_exec=1,
            lag_size_override=6,
        )
        exec_gs.execution()

        saved = generics.open_saved_result(exec_gs.title)
        fitted_pipeline = saved[0]["experiment"].model
        assert fitted_pipeline.n_features_in_ == 6
