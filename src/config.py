from pathlib import Path

def encontrar_raiz_projeto():
    diretorio_atual = Path(__file__).parent
    while diretorio_atual != diretorio_atual.parent:
        if (diretorio_atual / "setup.py").exists():
            return diretorio_atual
        diretorio_atual = diretorio_atual.parent
    return None

ROOT_PATH = str(encontrar_raiz_projeto())

RAW_DATA_PATH = ROOT_PATH + '/data/raw/'
MODEL_DATA_PATH = ROOT_PATH + '/data/result/'

TEST_SIZE = 0.2
VAL_SIZE = 0.2