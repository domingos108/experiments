"""
`GridSearch.execution()` deve honrar `force`/`file_exists` da MESMA forma que
`grid_seach_multiple_bases()` ja faz -- quando `force=False` e o `.pkl` de
resultado ja existe, a serie e PULADA (nao re-executada).

Regressao real: os notebooks de FS chamam `GridSearch(...).execution()` direto
num laco por serie (nao via `grid_seach_multiple_bases`), e `execution()` NUNCA
checava `file_exists` -- entao `force=False` era no-op ali. Adicionar uma serie
a `fs_series_list` e rodar "Run All" re-executava TODAS as series, incluindo as
ja validadas, com novo sorteio estocastico (MLP `model_exec=10` sem seed).
"""

import os

import pytest

import config
from model import generics, grid_search_exp
from tests.model.conftest import FakeModelExpValOnly


class _CountingModelExp(FakeModelExpValOnly):
    """FakeModelExpValOnly + contador de classe de quantas vezes fit_predict
    rodou -- para distinguir 'pulou' de 're-executou' por comportamento."""

    fit_predict_calls = 0

    def fit_predict(self):
        type(self).fit_predict_calls += 1
        super().fit_predict()


@pytest.fixture(autouse=True)
def _reset_counter():
    _CountingModelExp.fit_predict_calls = 0
    yield


def _build(tmp_path, monkeypatch, force):
    from sklearn.dummy import DummyRegressor

    monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
    return grid_search_exp.GridSearch(
        _CountingModelExp,
        DummyRegressor(strategy="constant", constant=0.0),
        {"constant": [1.0]},
        "fake_force_skip",
        "airlines.txt",
        "testmodel",
        force=force,
        normalize=True,
        experiment_params={"horizon": 1, "diff_kpss": False},
        model_exec=1,
    )


class TestExecutionHonorsForce:
    def test_force_false_skips_when_pkl_already_exists(self, tmp_path, monkeypatch):
        first = _build(tmp_path, monkeypatch, force=False)
        first.execution()
        assert generics.file_exists(first.title)
        first_mtime = os.path.getmtime(first.title)

        _CountingModelExp.fit_predict_calls = 0
        second = _build(tmp_path, monkeypatch, force=False)
        second.execution()

        assert _CountingModelExp.fit_predict_calls == 0  # nao re-executou
        assert os.path.getmtime(second.title) == first_mtime  # .pkl intacto

    def test_force_false_runs_when_pkl_absent(self, tmp_path, monkeypatch):
        gs = _build(tmp_path, monkeypatch, force=False)
        assert not generics.file_exists(gs.title)

        gs.execution()

        assert _CountingModelExp.fit_predict_calls > 0
        assert generics.file_exists(gs.title)

    def test_zero_byte_pkl_is_not_treated_as_done(self, tmp_path, monkeypatch):
        """Achado de code-review: um .pkl truncado a 0 bytes (queda no meio de
        save_result) nao deve fazer execution() pular para sempre com force=False."""
        gs = _build(tmp_path, monkeypatch, force=False)
        generics.create_path_if_not_exists(gs.fold)
        open(gs.title, "wb").close()  # arquivo vazio
        assert os.path.getsize(gs.title) == 0

        gs.execution()

        assert _CountingModelExp.fit_predict_calls > 0  # rodou de verdade
        assert os.path.getsize(gs.title) > 0  # sobrescreveu com resultado real

    def test_force_true_reexecutes_even_when_pkl_exists(self, tmp_path, monkeypatch):
        first = _build(tmp_path, monkeypatch, force=True)
        first.execution()
        assert generics.file_exists(first.title)

        _CountingModelExp.fit_predict_calls = 0
        second = _build(tmp_path, monkeypatch, force=True)
        second.execution()

        assert _CountingModelExp.fit_predict_calls > 0  # comportamento antigo preservado


class TestGridSeachMultipleBasesStillSkips:
    """Regressao: apos mover o skip para execution(), o wrapper de multiplas
    bases continua pulando series ja feitas quando force=False."""

    def test_multiple_bases_force_false_skips_existing(self, tmp_path, monkeypatch):
        from sklearn.dummy import DummyRegressor

        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        common = dict(
            fit_predict_class=_CountingModelExp,
            model=DummyRegressor(strategy="constant", constant=0.0),
            normalize=True,
            model_parameters={"constant": [1.0]},
            experiment_params={"horizon": 1, "diff_kpss": False},
            model_exec=1,
            model_name="testmodel",
            experiment_id="fake_multi_skip",
        )

        grid_search_exp.grid_seach_multiple_bases(**common, force=True)
        assert _CountingModelExp.fit_predict_calls > 0

        _CountingModelExp.fit_predict_calls = 0
        grid_search_exp.grid_seach_multiple_bases(**common, force=False)
        assert _CountingModelExp.fit_predict_calls == 0  # tudo ja existe -> pulou
