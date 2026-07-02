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
