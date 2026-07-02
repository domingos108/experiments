"""
Teste de regressao do bug de separador de caminho em metrics.py.

open_fold_result() extrai o nome da serie a partir do caminho retornado por
glob(). No Linux/Mac, glob() sempre usa '/'. No Windows, glob() insere '\\'
entre o diretorio e o nome do arquivo mesmo quando o diretorio-base foi
construido com '/' (caso de config.MODEL_DATA_PATH, que mistura os dois
estilos) -- pth.split('/')[-1] entao falha silenciosamente, extraindo o
nome da PASTA (ou parte dela) em vez do nome real da serie.

Isso quebrava metrics.open_fold_result()/calculate_metrics_v2.ipynb no
Windows para QUALQUER experiment_id, incluindo os 5 baselines originais
(ver relatorio da Tarefa 2, item 5).
"""

from metrics.metrics import extract_series_name_from_path


class TestExtractSeriesNameFromPath:
    def test_unix_style_path(self):
        pth = "/home/user/experiments/data/result/chamados/airlines_1amv1.pkl"
        assert extract_series_name_from_path(pth) == "airlines"

    def test_windows_style_path_pure_backslash(self):
        pth = r"C:\Projetos\experiments\data\result\chamados\airlines_1amv1.pkl"
        assert extract_series_name_from_path(pth) == "airlines"

    def test_windows_style_mixed_separators_reproducing_the_real_bug(self):
        """Reproduz exatamente o caminho que glob() produz neste projeto no
        Windows: base construida com '/' (config.MODEL_DATA_PATH), glob()
        inserindo '\\' antes do nome do arquivo. A logica antiga
        (pth.split('/')[-1]) extraia 'chamados' em vez de 'airlines' --
        exatamente o bug relatado na Tarefa 2."""
        pth = (
            r"C:\Projetos\mestrado_codigos\experiments"
            "/data/result/chamados_v2_fs_ftest"
            r"\airlines_1amv1.pkl"
        )
        assert extract_series_name_from_path(pth) == "airlines"

    def test_series_name_with_underscore_in_experiment_id_does_not_leak_in(self):
        """experiment_id com underscore (ex: chamados_v2_fs_ftest) nao pode
        contaminar o nome da serie extraido -- essa era a manifestacao
        exata do bug."""
        pth = r"C:\...\data\result\chamados_v2_fs_ftest\sunspot_1amv1.pkl"
        assert extract_series_name_from_path(pth) == "sunspot"
