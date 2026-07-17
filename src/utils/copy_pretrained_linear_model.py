"""
copy_pretrained_linear_model.py
---------------------------------
Copia o .pkl de um modelo linear ja pre-treinado (tipicamente ARIMA) de um
experiment_id de origem para um experiment_id novo -- necessario porque
`Additive` (src/model/hybrid_system_exp.py, via `input_linear_info`) exige
que o .pkl do modelo linear exista sob o MESMO `experiment_id` da variante
de Feature Selection, mas o modelo linear em si nao muda entre variantes de
FS (mesma serie, mesmo ARIMA) -- copiar evita retreinar.

Design (Tarefa 3.2, achado de code-review): a celula de copia estava
duplicada em 4 notebooks, reimplementando a construcao de caminho
manualmente (`Path(config.MODEL_DATA_PATH) / experiment_id / f'{serie}_1arima.pkl'`)
e cravando o literal '1arima'. Esta funcao usa `generics.format_names()` --
o mesmo helper que `Additive`/`input_linear_info` usam para localizar esse
.pkl -- e recebe `linear_model_name` como parametro em vez de um literal,
lido de `experiment_params['linear_model_name']` no notebook chamador.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from model import generics


def copy_pretrained_linear_model(
    source_experiment_id: str,
    dest_experiment_id: str,
    series_list: list[str],
    linear_model_name: str,
) -> None:
    """
    Copia `<serie>_<linear_model_name>.pkl` de `source_experiment_id` para
    `dest_experiment_id`, para cada serie em `series_list`. Nunca retreina
    -- so copia o .pkl ja existente.

    Levanta `FileNotFoundError` com uma mensagem que identifica a serie e o
    experiment_id de origem se o .pkl nao existir la -- mais claro que o
    OSError generico que `shutil.copy` levantaria sozinho.
    """
    for base_name in series_list:
        _, src_title = generics.format_names(source_experiment_id, base_name, linear_model_name)
        dest_fold, dest_title = generics.format_names(dest_experiment_id, base_name, linear_model_name)

        src_path = Path(src_title)
        if not src_path.exists():
            raise FileNotFoundError(
                f"'{linear_model_name}' pre-treinado nao encontrado para '{base_name}' em "
                f"'{source_experiment_id}': {src_path}. Rode o notebook/script que gera esse "
                f"modelo linear antes de copiar para '{dest_experiment_id}'."
            )

        generics.create_path_if_not_exists(dest_fold)
        shutil.copy(src_path, dest_title)
        print(f"{base_name.split('.')[0]}: {src_path} -> {dest_title}")
