"""
Testes de src/utils/copy_pretrained_linear_model.py (Tarefa 3.2).

Extraido do code-review: a celula de copia do ARIMA, duplicada em 4
notebooks, reimplementava a construcao de caminho manualmente e cravava o
literal '1arima' em vez de usar `generics.format_names()` (o mesmo helper
que `Additive`/`input_linear_info` usam para localizar o .pkl) e
`experiment_params['linear_model_name']` (ja definido na celula de
configuracao). Esta funcao corrige os dois problemas e fica em um unico
lugar, chamada pelos 4 notebooks.
"""

from pathlib import Path

import pytest

import config
from model import generics
from utils.copy_pretrained_linear_model import copy_pretrained_linear_model


class TestCopyPretrainedLinearModel:
    def test_copies_pkl_for_each_series_using_format_names_convention(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("chamados", "airlines.txt", "1arima")
        generics.save_result(fold, title, [{"fake": "arima_result"}])

        copy_pretrained_linear_model(
            source_experiment_id="chamados",
            dest_experiment_id="chamados_v4_fs_ftest",
            series_list=["airlines.txt"],
            linear_model_name="1arima",
        )

        _, expected_dest = generics.format_names("chamados_v4_fs_ftest", "airlines.txt", "1arima")
        assert Path(expected_dest).exists()
        assert generics.open_saved_result(expected_dest) == [{"fake": "arima_result"}]

    def test_raises_clear_filenotfounderror_when_source_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")

        with pytest.raises(FileNotFoundError, match="austres"):
            copy_pretrained_linear_model(
                source_experiment_id="chamados",
                dest_experiment_id="chamados_v4_fs_ftest",
                series_list=["austres.txt"],
                linear_model_name="1arima",
            )

    def test_uses_linear_model_name_parameter_not_a_hardcoded_value(self, tmp_path, monkeypatch):
        """Prova mecanica de que o nome do modelo linear vem do parametro,
        nao de um literal '1arima' cravado -- reproduz o achado do
        code-review sobre experiment_params['linear_model_name']."""
        monkeypatch.setattr(config, "MODEL_DATA_PATH", str(tmp_path) + "/")
        fold, title = generics.format_names("chamados", "sunspot.txt", "1outro_modelo_linear")
        generics.save_result(fold, title, [{"fake": "result"}])

        copy_pretrained_linear_model(
            source_experiment_id="chamados",
            dest_experiment_id="chamados_v4_fs_ftest",
            series_list=["sunspot.txt"],
            linear_model_name="1outro_modelo_linear",
        )

        _, expected_dest = generics.format_names("chamados_v4_fs_ftest", "sunspot.txt", "1outro_modelo_linear")
        assert Path(expected_dest).exists()
