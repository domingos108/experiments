"""
Teste do fallback aditivo `fs_lag_size` descrito no PLANO_ARQUITETURA.md
(Secao 1.3): GridSearch deve continuar resolvendo `lag_size` exatamente como
hoje quando a serie nao tem `fs_lag_size` definido em BASE_INFORMATION (os 5
baselines da Secao 3 do CLAUDE.md nao podem mudar de comportamento), e deve
usar o valor de `fs_lag_size` quando presente.
"""

import config
from model.grid_search_exp import resolve_lag_size


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

    def test_fs_lag_size_series_resolve_to_the_deliberately_configured_value(self):
        """As unicas series com fs_lag_size hoje sao as 2 usadas na Tarefa 2
        do PLANO_ARQUITETURA.md (integracao no arima_mlp.ipynb). Trava os
        valores exatos para que uma mudanca futura em config.py seja
        intencional e visivel, nao silenciosa."""
        assert resolve_lag_size(config.BASE_INFORMATION["sunspot.txt"]) == 30
        assert resolve_lag_size(config.BASE_INFORMATION["airlines.txt"]) == 20


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
