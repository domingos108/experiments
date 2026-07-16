"""
Laboratório Exploratório: Informação Mútua vs CCF (In-Sample)

Este script compara a capacidade da Informação Mútua (MI) e da 
Correlação Cruzada Linear (CCF) em detectar dependências não-lineares
entre a série real e as previsões lineares do ARIMA.

VERSÃO 2: Usa dados de TREINAMENTO (in-sample) para ter volume suficiente
de amostras para calcular MI com confiabilidade estatística.

Objetivo: Provar que a MI captura dependências que a CCF descarta,
justificando a arquitetura NoLiC proposta na dissertação.

Séries testadas: Sunspot, Lynx, Airlines
"""

import numpy as np
import pandas as pd
import os
import glob
from statsmodels.tsa.stattools import ccf
from sklearn.feature_selection import mutual_info_regression
from model.generics import open_saved_result


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Diretório base do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULT_BASE_PATH = os.path.join(BASE_DIR, 'data', 'result')

# Diretório para análises exploratórias
EXPLORATORY_PATH = os.path.join(RESULT_BASE_PATH, 'exploratory_analysis')

# Séries a testar com configurações de corte
SERIES_CONFIG = {
    'sunspot': {
        'test_size': 67,  # Khashei & Bijari (2011)
        'val_size': 15    # NoLiC baseline
    },
    'lynx': {
        'test_size': 14,  # Khashei & Bijari (2011)
        'val_size': 15    # NoLiC baseline
    },
    'airlines': {
        'test_size': 12,  # Santos Jr. (2024)
        'val_size': 15    # NoLiC baseline
    }
}

# ID do experimento NoLiC
EXPERIMENT_ID = 'nolic_baseline'

# Número máximo de lags a testar (fixo para todas as séries)
MAX_LAGS = 20


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def find_arima_file(experiment_id, serie_name):
    """
    Busca arquivo ARIMA usando glob (evita problemas com prefixos numéricos).
    
    Parâmetros
    ----------
    experiment_id : str
        ID do experimento (ex: 'nolic_baseline')
    serie_name : str
        Nome da série sem extensão (ex: 'sunspot')
    
    Retorna
    -------
    str ou None
        Caminho do arquivo encontrado ou None
    """
    pattern = os.path.join(RESULT_BASE_PATH, experiment_id, f"{serie_name}_*arima.pkl")
    arquivos = glob.glob(pattern)
    
    if not arquivos:
        return None
    
    # Ordenar por data de modificação e pegar o mais recente
    arquivos.sort(key=os.path.getmtime, reverse=True)
    return arquivos[0]


def extract_test_data(pkl_path):
    """
    Extrai dados de teste do objeto ARIMA salvo.
    
    Parâmetros
    ----------
    pkl_path : str
        Caminho do arquivo .pkl
    
    Retorna
    -------
    y_true : np.ndarray
        Série real de teste
    y_pred : np.ndarray
        Previsões lineares de teste
    """
    result_list = open_saved_result(pkl_path)
    
    # Estrutura: result_list[0]['experiment'].metrics_results
    experiment_obj = result_list[0]['experiment']
    metrics_results = experiment_obj.metrics_results
    
    # Extrair dados de teste
    # A série original de teste (target real)
    test_predict = metrics_results['test_predict']
    
    # Para obter y_true, precisamos usar a previsão e as métricas
    # Alternativa: usar residual_series se disponível
    # Mas o mais direto é calcular através do tamanho do teste
    
    # Verificar se temos test_metrics para validar
    test_metrics = metrics_results.get('test_metrics', {})
    
    # Para obter y_true, vamos assumir que podemos extrair do próprio resultado
    # Vamos usar uma abordagem diferente: carregar a série original
    # Mas isso é complicado... vamos usar o que temos
    
    # Na verdade, vamos precisar carregar a série original
    # O melhor é retornar apenas test_predict e usar outra abordagem
    
    # Alternativa: usar linear_forecast e residuals se disponíveis
    # Vamos retornar test_predict por enquanto
    return test_predict


def calculate_ccf_scores(y_true, y_pred, max_lags):
    """
    Calcula scores de CCF (Correlação Cruzada Linear) para múltiplos lags.
    
    Parâmetros
    ----------
    y_true : array-like
        Série real
    y_pred : array-like
        Previsões do modelo
    max_lags : int
        Número máximo de lags a calcular
    
    Retorna
    -------
    ccf_scores : np.ndarray
        Valores absolutos de CCF para lags 1 até max_lags
    significance_threshold : float
        Limite de significância 95% (1.96 / sqrt(N))
    """
    N = len(y_true)
    significance_threshold = 1.96 / np.sqrt(N)
    
    # Calcular CCF
    ccf_values = ccf(y_true, y_pred, adjusted=False)
    
    # ccf retorna [lag_-N, ..., lag_0, ..., lag_N]
    # Queremos apenas lags positivos: [1, 2, ..., max_lags]
    mid_idx = len(ccf_values) // 2
    
    # Extrair lags positivos
    ccf_scores = []
    for lag in range(1, max_lags + 1):
        if mid_idx + lag < len(ccf_values):
            ccf_scores.append(abs(ccf_values[mid_idx + lag]))
        else:
            ccf_scores.append(0.0)
    
    return np.array(ccf_scores), significance_threshold


def calculate_mi_scores(y_true, y_pred, max_lags):
    """
    Calcula scores de Informação Mútua (MI) para múltiplos lags.
    
    Cria um DataFrame com colunas sendo shifts (atrasos) de y_pred,
    remove NaNs e calcula MI entre cada lag e y_true.
    
    Parâmetros
    ----------
    y_true : array-like
        Série real
    y_pred : array-like
        Previsões do modelo
    max_lags : int
        Número máximo de lags a criar
    
    Retorna
    -------
    mi_scores : np.ndarray
        Scores de MI para lags 1 até max_lags
    """
    # Criar DataFrame com shifts (lags) de y_pred
    df_lags = pd.DataFrame()
    
    for lag in range(1, max_lags + 1):
        df_lags[f'lag_{lag}'] = pd.Series(y_pred).shift(lag)
    
    # Remover primeiras max_lags linhas (que têm NaNs)
    df_lags_clean = df_lags.iloc[max_lags:].reset_index(drop=True)
    
    # Alinhar y_true (remover primeiras max_lags linhas também)
    y_true_aligned = y_true[max_lags:]
    
    # Verificar se ainda temos dados suficientes
    if len(y_true_aligned) < 5:
        print(f"    [AVISO] Dados insuficientes após alinhamento: {len(y_true_aligned)} amostras")
        return np.zeros(max_lags)
    
    # Calcular MI para cada lag
    mi_scores = mutual_info_regression(
        df_lags_clean, 
        y_true_aligned, 
        random_state=42
    )
    
    return mi_scores


def analyze_series(serie_name, serie_config):
    """
    Analisa uma série: calcula CCF e MI usando dados de TREINAMENTO (in-sample),
    exibe tabela comparativa e salva resultados em CSV.
    
    Parâmetros
    ----------
    serie_name : str
        Nome da série (ex: 'sunspot')
    serie_config : dict
        Configuração com test_size e val_size
    """
    print("\n" + "=" * 80)
    print(f"SÉRIE: {serie_name.upper()}")
    print("=" * 80)
    
    # Buscar arquivo ARIMA
    pkl_path = find_arima_file(EXPERIMENT_ID, serie_name)
    
    if pkl_path is None:
        print(f"[ERRO] Arquivo ARIMA não encontrado para {serie_name}")
        print(f"       Padrão buscado: {serie_name}_*arima.pkl")
        return None
    
    print(f"[OK] Arquivo encontrado: {os.path.basename(pkl_path)}")
    
    # ========================================================================
    # CARREGAMENTO DE DADOS DE TREINAMENTO (IN-SAMPLE)
    # ========================================================================
    try:
        # Extrair previsões do modelo ARIMA salvo
        result_list = open_saved_result(pkl_path)
        experiment_obj = result_list[0]['experiment']
        metrics_results = experiment_obj.metrics_results
        
        # Extrair previsões de TREINO (in-sample)
        y_train_pred = metrics_results['train_predict']
        
        print(f"[OK] Previsões de treino extraídas: {len(y_train_pred)} amostras")
        
        # Carregar série original completa para extrair y_train_true
        serie_file = f"{serie_name}.txt"
        
        from input.input import load_raw_data
        
        df_complete = load_raw_data(serie_file)
        ts_complete = df_complete['y'].values
        
        # Calcular índices de corte
        test_size = serie_config['test_size']
        val_size = serie_config['val_size']
        train_end_idx = -(test_size + val_size)
        
        # y_train_true: tudo até o início da validação
        if train_end_idx == 0:
            y_train_true = ts_complete[:]
        else:
            y_train_true = ts_complete[:train_end_idx]
        
        # Alinhar tamanhos (y_train_pred pode ser menor devido a lags)
        # Pegar os últimos N valores de y_train_true para alinhar
        if len(y_train_pred) < len(y_train_true):
            y_train_true = y_train_true[-len(y_train_pred):]
        elif len(y_train_pred) > len(y_train_true):
            y_train_pred = y_train_pred[-len(y_train_true):]
        
        print(f"[OK] Dados de treino alinhados: {len(y_train_true)} amostras")
        
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dados: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Verificar se temos dados suficientes para análise
    if len(y_train_true) < MAX_LAGS + 10:
        print(f"[ERRO] Dados insuficientes para análise com {MAX_LAGS} lags")
        print(f"       Necessário: mínimo {MAX_LAGS + 10} amostras")
        print(f"       Disponível: {len(y_train_true)} amostras")
        return None
    
    print(f"[INFO] Analisando {MAX_LAGS} lags com {len(y_train_true)} amostras")
    
    # ========================================================================
    # CÁLCULO DA CCF (VISÃO NOLIC)
    # ========================================================================
    print("\n[1/2] Calculando CCF (Correlação Cruzada Linear)...")
    
    ccf_scores, significance_threshold = calculate_ccf_scores(
        y_train_true, 
        y_train_pred, 
        MAX_LAGS
    )
    
    print(f"      Limite de significância 95%: {significance_threshold:.4f}")
    
    # ========================================================================
    # CÁLCULO DA MI (VISÃO DA DISSERTAÇÃO)
    # ========================================================================
    print("\n[2/2] Calculando MI (Informação Mútua)...")
    
    mi_scores = calculate_mi_scores(
        y_train_true, 
        y_train_pred, 
        MAX_LAGS
    )
    
    # ========================================================================
    # TABELA COMPARATIVA
    # ========================================================================
    print("\n" + "-" * 80)
    print("RESULTADOS COMPARATIVOS: CCF vs MI (DADOS DE TREINO)")
    print("-" * 80)
    
    # Criar DataFrame para exibição e exportação
    df_comparison = pd.DataFrame({
        'Lag': range(1, MAX_LAGS + 1),
        'CCF_Absoluta': ccf_scores,
        'Significativo_CCF': ['Sim' if abs_ccf > significance_threshold else 'Não' 
                               for abs_ccf in ccf_scores],
        'Score_MI': mi_scores
    })
    
    # Adicionar coluna de diferença (MI que a CCF não detecta)
    df_comparison['MI_Descartado_CCF'] = [
        'Sim' if (sig == 'Não' and mi > 0.01) else 'Não'
        for sig, mi in zip(df_comparison['Significativo_CCF'], df_comparison['Score_MI'])
    ]
    
    print(df_comparison.to_string(index=False))
    
    # ========================================================================
    # ESTATÍSTICAS RESUMIDAS
    # ========================================================================
    print("\n" + "-" * 80)
    print("ESTATÍSTICAS RESUMIDAS")
    print("-" * 80)
    
    n_sig_ccf = (df_comparison['Significativo_CCF'] == 'Sim').sum()
    n_descartado = (df_comparison['MI_Descartado_CCF'] == 'Sim').sum()
    
    print(f"Lags significativos pela CCF (95%): {n_sig_ccf}/{MAX_LAGS}")
    print(f"Lags com MI relevante descartados pela CCF: {n_descartado}/{MAX_LAGS}")
    
    if n_descartado > 0:
        print(f"\n✓ EVIDÊNCIA ENCONTRADA: MI detectou {n_descartado} lag(s) com dependência")
        print(f"  não-linear que a CCF linear descartou!")
        print(f"  Isso justifica a arquitetura NoLiC proposta na dissertação.")
    else:
        print(f"\n○ Nesta série, a CCF linear foi suficiente para capturar as dependências.")
    
    # ========================================================================
    # EXPORTAÇÃO EM CSV
    # ========================================================================
    print("\n" + "-" * 80)
    print("EXPORTANDO RESULTADOS")
    print("-" * 80)
    
    # Criar diretório se não existir
    os.makedirs(EXPLORATORY_PATH, exist_ok=True)
    
    # Nome do arquivo CSV
    csv_filename = f"mi_vs_ccf_{serie_name}.csv"
    csv_path = os.path.join(EXPLORATORY_PATH, csv_filename)
    
    # Salvar DataFrame
    df_comparison.to_csv(csv_path, index=False)
    
    print(f"[✓] Resultados salvos em: {csv_path}")
    
    print("=" * 80)
    
    return df_comparison


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Executa análise comparativa CCF vs MI para todas as séries.
    
    VERSÃO 2: Usa dados de TREINAMENTO (in-sample) para ter volume
    suficiente de amostras para calcular MI com confiabilidade estatística.
    """
    print("=" * 80)
    print("LABORATÓRIO EXPLORATÓRIO: INFORMAÇÃO MÚTUA vs CCF")
    print("VERSÃO 2: ANÁLISE COM DADOS DE TREINAMENTO (IN-SAMPLE)")
    print("=" * 80)
    print(f"\nObjetivo: Provar que a MI captura dependências não-lineares")
    print(f"          que a CCF linear (usada no NoLiC) descarta.")
    print(f"\nSéries testadas: {', '.join(SERIES_CONFIG.keys())}")
    print(f"Lags analisados: {MAX_LAGS} (fixo para todas as séries)")
    print(f"\nMelhoria: Usando dados de TREINO para ter volume suficiente!")
    print("=" * 80)
    
    # Resultados agregados
    results_summary = []
    
    # Analisar cada série
    for serie_name, serie_config in SERIES_CONFIG.items():
        try:
            df_result = analyze_series(serie_name, serie_config)
            
            if df_result is not None:
                # Calcular estatísticas resumidas
                n_sig_ccf = (df_result['Significativo_CCF'] == 'Sim').sum()
                n_descartado = (df_result['MI_Descartado_CCF'] == 'Sim').sum()
                
                results_summary.append({
                    'serie': serie_name,
                    'lags_sig_ccf': n_sig_ccf,
                    'lags_mi_descartado': n_descartado,
                    'percentual_descartado': (n_descartado / MAX_LAGS) * 100
                })
            
        except Exception as e:
            print(f"\n[ERRO FATAL] Falha ao analisar {serie_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # RESUMO GERAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMO GERAL DA ANÁLISE")
    print("=" * 80)
    
    if results_summary:
        df_summary = pd.DataFrame(results_summary)
        print("\nEstatísticas por Série:")
        print(df_summary.to_string(index=False))
        
        # Salvar resumo geral
        summary_path = os.path.join(EXPLORATORY_PATH, 'resumo_mi_vs_ccf.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"\n[✓] Resumo geral salvo em: {summary_path}")
    
    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 80)
    print("\nInterpretação:")
    print("- CCF (NoLiC): Detecta apenas correlações LINEARES")
    print("- MI (Proposta): Detecta dependências NÃO-LINEARES")
    print("- Coluna 'MI_Descartado_CCF': Lags que o NoLiC ignora mas que têm informação!")
    print(f"\nResultados salvos em: {EXPLORATORY_PATH}")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Análise cancelada pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO FATAL] {str(e)}")
        import traceback
        traceback.print_exc()
