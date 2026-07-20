"""
Testes de src/model/grid_search_exp.py::load_grid_search_history (Tarefa
3.4, Objetivo 3): utilitario de leitura que transforma `grid_search_history`
(persistido por GridSearch) num DataFrame -- uma linha por combinacao
testada, pronto para plotar "erro de validacao vs. hiperparametro" (ex. `k`).

Colocado em model/grid_search_exp.py (nao src/utils/) porque le um formato
que so GridSearch produz -- achado de code-review, Tarefa 3.4.
"""

from pathlib import Path

import pytest

import config
from model import generics
from model.grid_search_exp import load_grid_search_history
from tests.model.conftest import make_fake_grid_search


def _run_fake_grid_search(tmp_path, monkeypatch, model_parameters, **kwargs):
    exec_gs = make_fake_grid_search(tmp_path, monkeypatch, model_parameters, **kwargs)
    exec_gs.execution()
    return exec_gs.title


class TestLoadGridSearchHistory:
    def test_returns_one_row_per_combination_with_param_and_metric_columns(self, tmp_path, monkeypatch):
        title = _run_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0, 2.0, 3.0]})

        df = load_grid_search_history(Path(title))

        assert len(df) == 3
        assert set(df["constant"]) == {1.0, 2.0, 3.0}
        assert "val_metric_mean" in df.columns
        assert "val_metric_std" in df.columns

    def test_val_metric_mean_matches_the_tested_value(self, tmp_path, monkeypatch):
        """O double usa o proprio parametro como RMSE de validacao -- o
        DataFrame deve refletir isso exatamente, provando que a leitura nao
        embaralha combinacoes."""
        title = _run_fake_grid_search(tmp_path, monkeypatch, {"constant": [1.0, 2.0, 3.0]})

        df = load_grid_search_history(Path(title)).set_index("constant")

        for constant_value in [1.0, 2.0, 3.0]:
            assert df.loc[constant_value, "val_metric_mean"] == pytest.approx(constant_value)

    def test_raises_clear_error_when_pkl_has_no_grid_search_history(self, tmp_path, monkeypatch):
        """.pkl gerado com save_grid_history=False (ou anterior a Tarefa 3.4)
        nao tem a chave -- a funcao deve falhar com uma mensagem clara, nao
        um KeyError/IndexError cru."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("fake_no_history", "airlines.txt", "1testmodel")
        generics.save_result(fold, title, [{"experiment": None, "val_metric": 1.0}])

        with pytest.raises(ValueError, match="grid_search_history"):
            load_grid_search_history(Path(title))

    def test_raises_clear_error_when_grid_search_history_is_present_but_empty(self, tmp_path, monkeypatch):
        """Achado de code-review (angulo cross-file): grid_search_history=[]
        (presente mas vazio) precisa ser tratado como AUSENTE, nao como um
        DataFrame vazio silencioso."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("fake_empty_history", "airlines.txt", "1testmodel")
        generics.save_result(fold, title, [{"experiment": None, "val_metric": 1.0, "grid_search_history": []}])

        with pytest.raises(ValueError, match="grid_search_history"):
            load_grid_search_history(Path(title))
