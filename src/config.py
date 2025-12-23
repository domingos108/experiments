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

BASE_NAME_LIST = [
    #'recifeaccday.txt',
    #'recifeaccmonth.txt',
    #'recifeaccweek.txt'
#]
#[   
    'coloradoRiver.txt', 
    'sunspot.txt',
    'milk.txt', 
    'Unemployment.txt',
    'ausbee.txt',
    'austres.txt',
    'heartrate.txt',
    "ozon.txt",
    "pollution.txt",
    "redwine.txt",
    "gasoline.txt",
    "temperature.txt",
    "woolyrnq.txt",
    #'taylor.txt',
    'melbmin.txt'

]

LAG_SIZE_LIST = []


BASE_INFORMATION = {
    'coloradoRiver.txt': {"freq": "MS", 'm': 12 }, 
    'sunspot.txt': {"freq": "YE", 'm': 1 }, 
    'milk.txt': {"freq": "MS", 'm': 12 }, 
    'Unemployment.txt': {"freq": "MS", 'm': 1 }, 
    'ausbee.txt': {"freq": "MS", 'm': 1 }, 
    'austres.txt': {"freq": "QE", 'm': 1 }, 
    'heartrate.txt': {"freq": "MS", 'm': 1 }, 
    "ozon.txt": {"freq": "MS", 'm': 1 }, 
    "pollution.txt": {"freq": "MS", 'm': 1 }, 
    "redwine.txt": {"freq": "MS", 'm': 12 }, 
    "gasoline.txt": {"freq": "MS", 'm': 1 }, 
    "temperature.txt": {"freq": "MS", 'm': 1 }, 
    "woolyrnq.txt": {"freq": "QE", 'm': 4 }, 
    "melbmin.txt": {"freq": "D", 'm': 1 }, 
}