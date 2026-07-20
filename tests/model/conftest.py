"""
Fixtures compartilhadas para os testes de Feature Selection.

FS_DEV_SERIES e a constante isolada e explicita exigida pela Tarefa 1 do
PLANO_ARQUITETURA.md (Secao 3, item 1): as 4 series usadas para testar/
exercitar TimeSeriesFeatureSelector com dados reais, sem hardcode espalhado
pelos arquivos de teste. Adicionar uma 5a serie de desenvolvimento e uma
mudanca de uma linha aqui.
"""

import pytest

import config
from input.input import open_format_train_val_test

FS_DEV_SERIES = ["airlines", "austres", "coloradoRiver", "sunspot"]


def load_fs_dev_series_train_data(series_name, k_lags=5):
    """
    Carrega uma das series de FS_DEV_SERIES ja janelada em lags (X_train,
    y_train), usando o mesmo pipeline de producao (input.open_format_train_val_test).

    lag_size e passado diretamente (nao vem de config.BASE_INFORMATION) para
    manter o fixture independente de fs_lag_size/lag_size por serie -- essas
    chaves sao escopo da integracao real (Tarefa 2+), nao deste fixture de teste.
    """
    base_name = f"{series_name}.txt"
    exec_config = {
        "test_size": config.TEST_SIZE,
        "val_size": config.VAL_SIZE,
        "horizon": 1,
        "lag_size": k_lags,
        "diff_kpss": False,
        "normalize": False,
        "type_filter": None,
    }
    base_info = open_format_train_val_test(base_name, exec_config)
    X_train = base_info.df_train.drop(columns=["actual"]).values
    y_train = base_info.df_train["actual"].values
    return X_train, y_train


@pytest.fixture(params=FS_DEV_SERIES)
def fs_dev_series_train_data(request):
    """Parametrizada sobre as 4 series de FS_DEV_SERIES -- um teste que usa
    este fixture roda automaticamente uma vez por serie."""
    return load_fs_dev_series_train_data(request.param)


class FakeModelExpValOnly:
    """Test double de `model_class_exp` (compartilhado entre
    test_grid_search_history.py e test_read_grid_search_history.py -- achado
    de code-review da Tarefa 3.4, os dois arquivos tinham copias quase
    identicas que ja haviam divergido). Nunca treina nada de verdade -- usa
    o proprio parametro testado (`model.constant`) como RMSE de validacao,
    com um `test_metrics` SENTINELA fixo e sempre DIFERENTE do valor de
    validacao. Prova mecanica (nao estatistica) de que o historico do Grid
    Search so pode vir de `val_metrics`."""

    TEST_SENTINEL = 999999.0

    def __init__(self, model, experiment_id, base_name, model_name, force, normalize, experiment_params):
        self.model = model
        self.metrics_results = None

    def fit_predict(self):
        val_rmse = float(self.model.constant)
        self.metrics_results = {
            "val_metrics": {"RMSE": val_rmse},
            "test_metrics": {"RMSE": self.TEST_SENTINEL},
        }


def make_fake_grid_search(tmp_path, monkeypatch, model_parameters, model_exec=2, save_grid_history=True):
    """Constroi (sem executar) um GridSearch de teste, usando
    `FakeModelExpValOnly` sobre `DummyRegressor(strategy='constant')` --
    chame `.execution()` no teste para rodar."""
    from sklearn.dummy import DummyRegressor

    from model import grid_search_exp

    monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

    return grid_search_exp.GridSearch(
        FakeModelExpValOnly,
        DummyRegressor(strategy="constant", constant=0.0),
        model_parameters,
        "fake_experiment",
        "airlines.txt",
        "testmodel",
        force=True,
        normalize=True,
        experiment_params={"horizon": 1, "diff_kpss": False},
        model_exec=model_exec,
        save_grid_history=save_grid_history,
    )
