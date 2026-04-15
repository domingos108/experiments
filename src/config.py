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
    #'majaboataosamu.txt',
    #'maolindasamu.txt',
    #'mapaulistasamu.txt',
    #'marecifesamu.txt',
    #'marecacc.txt',

    #'consumocoformated.txt',
    #'windspeedrecife.txt',
    #'windspeednatal.txt',
    #'windspeedfortaleza.txt',
    #'windspeedsaoluis.txt',
    #'irradiancesalvador.txt',
    #'irradiancefortaleza.txt',
    #'irradiancefloripa.txt',
    #'irradiancesp.txt',

    #'coloradoRiver.txt', 
    #'sunspot.txt',
    #'milk.txt', 
    #'Unemployment.txt',
    #'ausbee.txt',
    #'austres.txt',
    #'heartrate.txt',
    #"ozon.txt",
    #"pollution.txt",
    #"redwine.txt",
    #"gasoline.txt",
    #"temperature.txt",
    #"woolyrnq.txt",
    #'melbmin.txt',

   
    #'consumoneformated.txt',
    #'consumonoformated.txt',
    #'consumosdformated.txt',
    #'consumosulformated.txt'

    #'irradiancecabrobo.txt',
    #'irradianceouricuri.txt',
    #'irradiancesalgueiro.txt'

    ##'irradiacaoouricuri2024.txt', 
    #'irradiacaocabrobo2024.txt', 
    #'irradiacaosalgueiro2024.txt', 
    #'irradiancecastelopiaui.txt', 
    #'irradiancepicospi.txt', 
    #'irradiancevalencapiaui.txt' 

    #'windspeedcampinagrandehour.txt',
    'windspeedcaruaruhour.txt',
    'windspeedcastelodopiauihour.txt', 
    'windspeedmaceiohour.txt' ,
    'windspeedsalvadorhour.txt',

]

LAG_SIZE_LIST = []


BASE_INFORMATION = {
    'windspeedcampinagrandehour.txt': {"freq": "D", 'm': 1, 'lag_size': 24}, 
    'windspeedcaruaruhour.txt': {"freq": "D", 'm': 1, 'lag_size': 24}, 
    'windspeedcastelodopiauihour.txt': {"freq": "D", 'm': 1, 'lag_size': 24}, 
    'windspeedmaceiohour.txt': {"freq": "D", 'm': 1, 'lag_size': 24}, 
    'windspeedsalvadorhour.txt': {"freq": "D", 'm': 1, 'lag_size': 24}, 

    'irradiancecabrobo.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    'irradianceouricuri.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    'irradiancesalgueiro.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 

    'irradiacaocabrobo2024.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    'irradiacaoouricuri2024.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    'irradiacaosalgueiro2024.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    'irradiancecastelopiaui.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    'irradiancepicospi.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    'irradiancevalencapiaui.txt': {"freq": "D", 'm': 13, 'lag_size': 13}, 
    

    'marecacc.txt': {"freq": "D", 'm': 7 , 'lag_size': 7}, 
 
    'majaboataosamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},
    'maolindasamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},
    'mapaulistasamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},
    'marecifesamu.txt': {"freq": "D", 'm': 7 , 'lag_size': 7},

    'windspeedrecife.txt': {"freq": "MS", 'm': 12, 'lag_size': 12 }, 
    'windspeednatal.txt': {"freq": "MS", 'm': 12, 'lag_size': 12  }, 
    'windspeedfortaleza.txt': {"freq": "MS", 'm': 12 , 'lag_size': 12 }, 
    'windspeedsaoluis.txt': {"freq": "MS", 'm': 12 , 'lag_size': 12 },

    'irradiancesalvador.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto' }, 
    'irradiancefortaleza.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto'  }, 
    'irradiancefloripa.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto'  }, 
    'irradiancesp.txt': {"freq": "D", 'm': 7, 'lag_size': 'auto'  }, 

    'coloradoRiver.txt': {"freq": "MS", 'm': 12, 'lag_size': 'auto' }, 
    'sunspot.txt': {"freq": "YE", 'm': 1, 'lag_size': 'auto'  }, 
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