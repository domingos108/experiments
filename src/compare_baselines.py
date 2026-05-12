"""
Script para comparar baselines híbridos de Zhang (2003) e Khashei & Bijari (2011).

Extrai métricas MSE e MAD das execuções salvas e gera tabela comparativa.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Caminho base para resultados
RESULT_BASE_PATH = 'data/result'
COMPARISON_OUTPUT_PATH = 'data/result/comparisons'

# Séries para comparação (conforme literatura)
SERIES_LIST = ['sunspot.txt', 'lynx.txt', 'airlines.txt']

# Configuração dos modelos a comparar
MODELS_CONFIG = {
    'Zhang 2003': {
        'experiment_id': 'zhang_baseline',  # Ajuste conforme seu experiment_id do Zhang
        'model_name': 'amv1',                # Additive MLP v1
        'label': 'Zhang 2003'
    },
    'Khashei & Bijari 2011': {
        'experiment_id': 'kb_baseline',
        'model_name': 'kb_mlp',
        'label': 'Khashei & Bijari 2011'
    },
    # Preparado para adicionar NoLiCHybrid (Santos Jr) na próxima etapa
    # 'Santos Jr 2024': {
    #     'experiment_id': 'nolic_baseline',
    #     'model_name': 'nolic_mlp',
    #     'label': 'Santos Jr 2024'
    # }
}

# Renomear MAE para MAD na tabela final
METRIC_LABELS = {
    'MSE': 'MSE',
    'MAE': 'MAD'  # Convenção da literatura: MAE = MAD
}


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def build_pkl_path(experiment_id, base_name, model_name):
    """
    Constrói o caminho do arquivo .pkl seguindo o padrão do framework.
    
    Padrão: data/result/{experiment_id}/{serie_sem_extensao}_{model_name}.pkl
    
    Parâmetros
    ----------
    experiment_id : str
        ID do experimento
    base_name : str
        Nome da série (ex: 'sunspot.txt')
    model_name : str
        Nome do modelo (ex: 'kb_mlp', 'amv1')
    
    Retorna
    -------
    str
        Caminho completo do arquivo .pkl
    """
    # Remover extensão .txt do nome da série
    serie_name = base_name.replace('.txt', '')
    
    # Construir nome do arquivo
    filename = f"{serie_name}_{model_name}.pkl"
    
    # Construir caminho completo
    pkl_path = os.path.join(RESULT_BASE_PATH, experiment_id, filename)
    
    return pkl_path


def load_model_metrics(experiment_id, base_name, model_name):
    """
    Carrega métricas de teste de um modelo salvo usando pickle.
    
    Parâmetros
    ----------
    experiment_id : str
        ID do experimento
    base_name : str
        Nome da série (ex: 'sunspot.txt')
    model_name : str
        Nome do modelo (ex: 'kb_mlp', 'amv1')
    
    Retorna
    -------
    dict ou None
        Dicionário com métricas de teste ou None se não encontrado
    """
    # Construir caminho do arquivo
    pkl_path = build_pkl_path(experiment_id, base_name, model_name)
    
    # Verificar se arquivo existe
    if not os.path.exists(pkl_path):
        print(f"[AVISO] Arquivo não encontrado: {pkl_path}")
        return None
    
    try:
        # Carregar arquivo .pkl
        with open(pkl_path, 'rb') as f:
            results_list = pickle.load(f)
        
        # Estrutura esperada: [{'experiment': ResultExp, 'val_metric': float}, ...]
        if not results_list or len(results_list) == 0:
            print(f"[AVISO] Arquivo vazio: {pkl_path}")
            return None
        
        # Pegar primeira execução
        first_result = results_list[0]
        
        # Acessar objeto experiment
        if 'experiment' in first_result:
            experiment_obj = first_result['experiment']
            
            # Acessar metrics_results
            if hasattr(experiment_obj, 'metrics_results'):
                metrics_results = experiment_obj.metrics_results
                
                # Extrair métricas de teste
                if 'test_metrics' in metrics_results:
                    test_metrics = metrics_results['test_metrics']
                    return test_metrics
                else:
                    print(f"[AVISO] Chave 'test_metrics' não encontrada em: {pkl_path}")
                    return None
            else:
                print(f"[AVISO] Atributo 'metrics_results' não encontrado em: {pkl_path}")
                return None
        else:
            print(f"[AVISO] Estrutura inesperada (chave 'experiment' ausente): {pkl_path}")
            return None
            
    except Exception as e:
        print(f"[ERRO] Falha ao carregar {pkl_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def extract_metrics_for_series(base_name, models_config):
    """
    Extrai métricas de todos os modelos para uma série específica.
    
    Parâmetros
    ----------
    base_name : str
        Nome da série
    models_config : dict
        Configuração dos modelos
    
    Retorna
    -------
    dict
        Dicionário com métricas: {modelo_metrica: valor}
    """
    row_data = {}
    
    for model_key, config in models_config.items():
        experiment_id = config['experiment_id']
        model_name = config['model_name']
        label = config['label']
        
        # Carregar métricas
        test_metrics = load_model_metrics(experiment_id, base_name, model_name)
        
        if test_metrics is None:
            # Preencher com NaN se não encontrado
            row_data[f"{label}_MSE"] = np.nan
            row_data[f"{label}_MAD"] = np.nan
        else:
            # Extrair RMSE e MAE
            rmse = test_metrics.get('RMSE', np.nan)
            mae = test_metrics.get('MAE', np.nan)
            
            # Calcular MSE a partir de RMSE
            if not np.isnan(rmse):
                mse = rmse ** 2
            else:
                mse = np.nan
            
            # MAD = MAE (convenção da literatura)
            mad = mae
            
            # Armazenar métricas
            row_data[f"{label}_MSE"] = mse
            row_data[f"{label}_MAD"] = mad
    
    return row_data


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def generate_comparison_table():
    """
    Gera tabela comparativa dos modelos híbridos.
    
    Retorna
    -------
    pd.DataFrame
        Tabela com séries como índice e métricas por modelo como colunas
    """
    print("=" * 80)
    print("COMPARAÇÃO DE BASELINES HÍBRIDOS")
    print("=" * 80)
    print(f"\nSéries: {', '.join(SERIES_LIST)}")
    print(f"Modelos: {', '.join([cfg['label'] for cfg in MODELS_CONFIG.values()])}")
    print(f"Métricas: MSE, MAD")
    print("-" * 80)
    
    # Coletar dados para todas as séries
    data_rows = []
    
    for base_name in SERIES_LIST:
        print(f"\n[CARREGANDO] {base_name}...")
        
        row_data = extract_metrics_for_series(base_name, MODELS_CONFIG)
        
        # Adicionar nome da série
        row_data['Serie'] = base_name.replace('.txt', '')
        
        data_rows.append(row_data)
    
    # Criar DataFrame
    df = pd.DataFrame(data_rows)
    
    # Reordenar colunas: Serie primeiro, depois métricas agrupadas por modelo
    serie_col = ['Serie']
    metric_cols = [col for col in df.columns if col != 'Serie']
    df = df[serie_col + metric_cols]
    
    # Definir índice como Serie
    df = df.set_index('Serie')
    
    return df


def print_comparison_table(df):
    """
    Imprime tabela de comparação formatada.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        Tabela de comparação
    """
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA - MÉTRICAS DE TESTE")
    print("=" * 80)
    print()
    
    # Configurar pandas para exibição limpa
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print(df.to_string())
    print()
    print("=" * 80)
    
    # Estatísticas adicionais
    print("\nESTATÍSTICAS RESUMIDAS")
    print("-" * 80)
    
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col:<35} Média: {mean_val:>8.4f} | Std: {std_val:>8.4f}")
    
    print("=" * 80)


def export_to_csv(df, filename='comparison_baseline.csv'):
    """
    Exporta tabela de comparação para CSV em pasta organizada.
    
    Parâmetros
    ----------
    df : pd.DataFrame
        Tabela de comparação
    filename : str
        Nome do arquivo de saída
    """
    try:
        # Criar pasta de comparações se não existir
        os.makedirs(COMPARISON_OUTPUT_PATH, exist_ok=True)
        
        # Construir caminho completo
        output_path = os.path.join(COMPARISON_OUTPUT_PATH, filename)
        
        # Salvar CSV
        df.to_csv(output_path)
        
        print(f"\n[EXPORTADO] Tabela salva em: {output_path}")
        
    except Exception as e:
        print(f"\n[ERRO] Falha ao exportar CSV: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Função principal de comparação."""
    try:
        # Gerar tabela de comparação
        df_comparison = generate_comparison_table()
        
        # Imprimir tabela formatada
        print_comparison_table(df_comparison)
        
        # Exportar para CSV em pasta organizada
        export_to_csv(df_comparison, 'comparison_baseline.csv')
        
        print("\n[CONCLUIDO] Comparação de baselines concluída com sucesso!")
        
        return df_comparison
        
    except Exception as e:
        print(f"\n[ERRO FATAL] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Execução cancelada pelo usuário.")
        sys.exit(1)
