"""
Teste do fallback aditivo `fs_lag_size` descrito no PLANO_ARQUITETURA.md
(Secao 1.3): GridSearch deve continuar resolvendo `lag_size` exatamente como
hoje quando a serie nao tem `fs_lag_size` definido em BASE_INFORMATION (os 5
baselines da Secao 3 do CLAUDE.md nao podem mudar de comportamento), e deve
usar o valor de `fs_lag_size` quando presente.

Tarefa 3.1 (PLANO_ARQUITETURA.md Secao 1.5): a chave `fs_lag_size` foi
REMOVIDA de BASE_INFORMATION para as 4 series de FS_DEV_SERIES (decisao do
orientador -- nao mais um valor profundo manual por serie). O mecanismo de
fallback em si (resolve_lag_size) nao mudou; o que mudou e que agora nenhuma
serie usa o ramo `fs_lag_size`, entao as 4 series de FS_DEV_SERIES caem no
mesmo `lag_size='auto'` do baseline -- por isso TestFsDevSeriesDfTrainParity
prova, com dados reais, que o df_train resultante e identico ao do baseline
sem seletor.
"""

import numpy as np
import pandas as pd
import pytest

import config
from input.input import open_format_train_val_test
from model.grid_search_exp import resolve_lag_size
from tests.model.conftest import FS_DEV_SERIES


class TestResolveLagSizeRegression:
    def test_matches_current_lag_size_for_every_series_without_fs_lag_size(self):
        """Regressao explicita dos 5 baselines: para toda serie de
        BASE_INFORMATION que NAO tem fs_lag_size definido, o valor resolvido
        deve continuar bit-a-bit identico ao lag_size de sempre. Series com
        fs_lag_size (uso deliberado dos experimentos de Feature Selection,
        ex. sunspot.txt/airlines.txt) sao verificadas em teste proprio, nao
        aqui -- para elas mudar de valor eh o comportamento esperado."""
        for base_name, base_info in config.BASE_INFORMATION.items():
            if "fs_lag_size" in base_info:
                continue
            assert resolve_lag_size(base_info) == base_info["lag_size"]

    @pytest.mark.parametrize("series_name", FS_DEV_SERIES)
    def test_fs_dev_series_have_no_fs_lag_size_key_after_reversal(self, series_name):
        """Tarefa 3.1: a chave fs_lag_size foi removida (nao substituida por
        um literal 'auto') para as 4 series de FS_DEV_SERIES -- resolve_lag_size
        deve cair no fallback por AUSENCIA da chave, nao por coincidencia de
        valor. Trava isso para que uma reintroducao futura de fs_lag_size
        seja intencional e visivel, nao silenciosa."""
        base_info = config.BASE_INFORMATION[f"{series_name}.txt"]
        assert "fs_lag_size" not in base_info
        assert resolve_lag_size(base_info) == base_info["lag_size"] == "auto"


class TestResolveLagSizeFallbackActive:
    def test_uses_fs_lag_size_when_present(self):
        base_info = {"freq": "YE", "m": 1, "lag_size": "auto", "fs_lag_size": 30}

        assert resolve_lag_size(base_info) == 30

    def test_falls_back_to_lag_size_when_fs_lag_size_absent(self):
        base_info = {"freq": "MS", "m": 12, "lag_size": 12}

        assert resolve_lag_size(base_info) == 12

    def test_uses_fs_lag_size_even_when_lag_size_key_is_absent(self):
        """fs_lag_size deve ter prioridade de verdade -- inclusive quando
        lag_size nem existe na entrada -- e nao apenas quando ambas as
        chaves estao presentes."""
        base_info = {"freq": "YE", "m": 1, "fs_lag_size": 30}

        assert resolve_lag_size(base_info) == 30


class TestFsDevSeriesDfTrainParity:
    """Prova de regressao pedida explicitamente na Tarefa 3.1: com a chave
    fs_lag_size removida (fallback ativo), o df_train resultante para as 4
    series de FS_DEV_SERIES precisa ser IDENTICO ao do baseline sem seletor
    (lag_size='auto' direto, sem passar por resolve_lag_size). Usa dados
    reais via input.open_format_train_val_test -- mesma funcao usada em
    producao por Additive/SKlearnModel -- nao dados sinteticos."""

    def _open_with_lag_size(self, series_name, lag_size):
        base_name = f"{series_name}.txt"
        exec_config = {
            "test_size": config.TEST_SIZE,
            "val_size": config.VAL_SIZE,
            "horizon": 1,
            "lag_size": lag_size,
            "diff_kpss": False,
            "normalize": False,
            "type_filter": None,
        }
        return open_format_train_val_test(base_name, exec_config)

    @pytest.mark.parametrize("series_name", FS_DEV_SERIES)
    def test_df_train_matches_baseline_exactly(self, series_name):
        base_name = f"{series_name}.txt"
        resolved_lag_size = resolve_lag_size(config.BASE_INFORMATION[base_name])

        fs_path_result = self._open_with_lag_size(series_name, resolved_lag_size)
        baseline_result = self._open_with_lag_size(series_name, "auto")

        pd.testing.assert_frame_equal(fs_path_result.df_train, baseline_result.df_train)
        pd.testing.assert_frame_equal(fs_path_result.df_val, baseline_result.df_val)
        pd.testing.assert_frame_equal(fs_path_result.df_test, baseline_result.df_test)
        assert fs_path_result.lag_size_formated == baseline_result.lag_size_formated
