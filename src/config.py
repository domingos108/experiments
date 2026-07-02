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

TEST_SIZE = 0.1
VAL_SIZE = 0.1

BASE_NAME_LIST = [
    # --- Baseline clássica (17 séries) ---
    'airlines.txt',
    #'ausbee.txt',
    'austres.txt',
    'coloradoRiver.txt',
    #'gasoline.txt',
    #'heartrate.txt',
    #'lakeerie.txt',
    #'lynx.txt',
    #'milk.txt',
    #'ozon.txt',
    #'pollution.txt',
    #'redwine.txt',
    'sunspot.txt',
    #'taylor.txt',
    #'temperature.txt',
    #'Unemployment.txt',
    #'woolyrnq.txt',
]

LAG_SIZE_LIST = []


# Chave opcional 'fs_lag_size': usada apenas pelos experimentos de Feature
# Selection (ver PLANO_ARQUITETURA.md, Secao 1.3). Quando presente, tem
# prioridade sobre 'lag_size' e expõe uma janela de lags mais profunda ao
# TimeSeriesFeatureSelector -- ex: {'lag_size': 12, 'fs_lag_size': 30}.
# Nao populada para nenhuma serie ainda: a decisao de quais series recebem
# lags profundos e do pesquisador, fora do escopo desta tarefa.
BASE_INFORMATION = {
    'marecacc.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},
 
    'majaboataosamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},
    'maolindasamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},
    'mapaulistasamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},
    'marecifesamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},

    'windspeedrecife.txt': {"freq": "MS", 'm': 12, 'lag_size': 12 }, 
    'windspeednatal.txt': {"freq": "MS", 'm': 12, 'lag_size': 12  }, 
    'windspeedfortaleza.txt': {"freq": "MS", 'm': 12 , 'lag_size': 12 }, 

    'irradiancesalvador.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto' }, 
    'irradiancefortaleza.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto'  }, 
    'irradiancefloripa.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto'  }, 
    'irradiancesp.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto'  }, 

    # --- Baseline clássica: entradas adicionadas ---
    'airlines.txt':     {"freq": "MS",  'm': 12, 'lag_size': 'auto', 'fs_lag_size': 20},   # provisório
    'lakeerie.txt':     {"freq": "MS",  'm': 12, 'lag_size': 'auto'},   # provisório
    'lynx.txt':         {"freq": "YE",  'm': 1,  'lag_size': 'auto'},   # provisório
    'taylor.txt':       {"freq": "MS",  'm': 12, 'lag_size': 'auto'},   # provisório

    'coloradoRiver.txt': {"freq": "MS", 'm': 12, 'lag_size': 'auto' }, 
    'sunspot.txt': {"freq": "YE", 'm': 1, 'lag_size': 'auto', 'fs_lag_size': 30  },
    'milk.txt': {"freq": "MS", 'm': 12, 'lag_size': 'auto'  }, 
    'Unemployment.txt': {"freq": "MS", 'm': 1, 'lag_size': 'auto'  }, 
    'ausbee.txt': {"freq": "MS", 'm': 1, 'lag_size': 'auto'  }, 
    'austres.txt': {"freq": "QE", 'm': 1, 'lag_size': 'auto'  }, 
    'heartrate.txt': {"freq": "MS", 'm': 1, 'lag_size': 'auto'  }, 
    "ozon.txt": {"freq": "MS", 'm': 1, 'lag_size': 'auto'  }, 
    "pollution.txt": {"freq": "MS", 'm': 1, 'lag_size': 'auto'  }, 
    "redwine.txt": {"freq": "MS", 'm': 12, 'lag_size': 'auto'  }, 
    "gasoline.txt": {"freq": "MS", 'm': 1, 'lag_size': 'auto'  }, 
    "temperature.txt": {"freq": "MS", 'm': 1, 'lag_size': 'auto'  }, 
    "woolyrnq.txt": {"freq": "QE", 'm': 4, 'lag_size': 'auto'  }, 
    "melbmin.txt": {"freq": "D", 'm': 1, 'lag_size': 'auto'  }, 

    'consumocoformated.txt': {"freq": "MS", 'm': 12, 'lag_size': 12  },
    'consumoneformated.txt': {"freq": "MS", 'm': 12, 'lag_size': 12  },
    'consumonoformated.txt': {"freq": "MS", 'm': 12, 'lag_size': 12  },
    'consumosdformated.txt': {"freq": "MS", 'm': 12, 'lag_size': 12 },
    'consumosulformated.txt': {"freq": "MS", 'm': 12, 'lag_size': 12  }
}