"""
Script para executar baseline de Zhang (2003).

Roda a arquitetura híbrida aditiva (ARIMA + MLP nos resíduos) em séries 
clássicas usadas no paper original.

Referência
----------
Zhang, G. P. (2003). "Time series forecasting using a hybrid ARIMA and 
neural network model." Neurocomputing, 50, 159-175.
"""

import sys
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
from model.hybrid_system_exp import Additive
import config

# Importar gerador de baselines lineares
from run_arima_base import generate_linear_baselines


# ============================================================================
# CONFIGURAÇÕES POR SÉRIE (test_size e topologia MLP conforme literatura)
# ============================================================================

SERIES_CONFIG = {
    'sunspot.txt': {
        'test_size': 67,         # Zhang (2003)
        'lag_size': 12,          # Padrão para séries anuais com ciclo de 11 anos
        'mlp_topology': (4,)     # Topologia especificada no paper
    },
    'lynx.txt': {
        'test_size': 14,         # Zhang (2003)
        'lag_size': 12,          # Padrão para séries anuais
        'mlp_topology': (5,)     # Topologia especificada no paper
    },
    'airlines.txt': {
        'test_size': 12,         # Santos Jr. (2024) usou 12 para séries mensais
        'lag_size': 12,          # Padrão para séries mensais (1 ano)
        'mlp_topology': (6,)     # Topologia especificada no paper
    }
}


# ============================================================================
# PARÂMETROS GLOBAIS DO EXPERIMENTO
# ============================================================================

# ID do experimento (reutilizar ARIMA já treinado no kb_baseline)
EXPERIMENT_ID = 'kb_baseline'

# Nome do modelo ARIMA pré-treinado (deve existir antes de rodar este script)
LINEAR_MODEL_NAME = '1arima'

# Parâmetros da rede neural
MLP_MAX_ITER = 1000
MLP_RANDOM_STATE = 42

# Validação: Zhang (2003) não usou conjunto de validação explícito
VAL_SIZE = 0

# Horizonte de previsão
HORIZON = 1

# Diferenciação e normalização
DIFF_KPSS = False
NORMALIZE = True


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def run_zhang_baseline():
    """
    Executa baseline de Zhang (2003) nas séries clássicas.
    
    Modelo Aditivo: y_t = L_t + N_t
    - L_t: Componente linear (ARIMA)
    - N_t: Componente não-linear (MLP nos resíduos ARIMA)
    """
    series_list = ['sunspot.txt', 'lynx.txt', 'airlines.txt']
    
    print("=" * 80)
    print("BASELINE ZHANG (2003) - MODELO ADITIVO")
    print("=" * 80)
    print(f"\nExperiment ID: {EXPERIMENT_ID}")
    print(f"Linear Model: {LINEAR_MODEL_NAME}")
    print(f"val_size: {VAL_SIZE}")
    print("-" * 80)
    
    # ========================================================================
    # ETAPA 1: GARANTIR EXISTÊNCIA DOS MODELOS ARIMA BASE
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 1/2: GERANDO MODELOS ARIMA BASE")
    print("=" * 80)
    print("Pipeline integrado: Treinando modelos lineares antes dos híbridos...")
    print("Isso garante que os resíduos e previsões ARIMA estejam disponíveis.")
    print("-" * 80)
    
    arima_results = generate_linear_baselines(
        experiment_id=EXPERIMENT_ID,
        series_config=SERIES_CONFIG,
        val_size=VAL_SIZE,
        diff_kpss=DIFF_KPSS,
        horizon=HORIZON,
        force=False,  # Não sobrescrever se já existir
        model_name=LINEAR_MODEL_NAME
    )
    
    # Verificar se todos os modelos ARIMA foram criados com sucesso
    arima_failures = [s for s, r in arima_results.items() if r['status'] not in ['OK', 'SKIP']]
    if arima_failures:
        print(f"\n[ERRO] Falha ao gerar modelos ARIMA para: {', '.join(arima_failures)}")
        print("Pipeline híbrido não pode continuar sem os modelos lineares base.")
        return
    
    print("\n[✓] Modelos ARIMA base prontos! Iniciando treinamento de modelos híbridos...")
    
    # ========================================================================
    # ETAPA 2: TREINAMENTO DOS MODELOS HÍBRIDOS ADITIVOS (ZHANG)
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 2/2: TREINANDO MODELOS HÍBRIDOS ZHANG (2003)")
    print("=" * 80)
    
    results_summary = []
    
    for base_name in series_list:
        print(f"\n[INICIO] Processando: {base_name}")
        
        # Verificar se série tem configuração
        if base_name not in SERIES_CONFIG:
            print(f"[ERRO] Configuração não encontrada para {base_name}")
            print(f"       Séries disponíveis: {list(SERIES_CONFIG.keys())}")
            continue
        
        # Carregar configuração específica da série
        serie_config = SERIES_CONFIG[base_name]
        test_size = serie_config['test_size']
        lag_size = serie_config['lag_size']
        mlp_topology = serie_config['mlp_topology']
        
        print(f"  test_size: {test_size}")
        print(f"  lag_size: {lag_size}")
        print(f"  MLP topology: {mlp_topology}")
        
        # Configurar parâmetros do experimento
        experiment_params = {
            'linear_model_name': LINEAR_MODEL_NAME,
            'test_size': test_size,
            'val_size': VAL_SIZE,
            'horizon': HORIZON,
            'lag_size': lag_size,
            'diff_kpss': DIFF_KPSS
        }
        
        # Instanciar modelo MLP com topologia específica da série
        mlp = MLPRegressor(
            hidden_layer_sizes=mlp_topology,
            max_iter=MLP_MAX_ITER,
            random_state=MLP_RANDOM_STATE,
            activation='logistic',
            solver='adam'
        )
        
        # Instanciar experimento Aditivo (Zhang)
        zhang_hybrid = Additive(
            model=mlp,
            experiment_id=EXPERIMENT_ID,
            base_name=base_name,
            model_name='zhang_mlp',
            normalize=NORMALIZE,
            experiment_params=experiment_params
        )
        
        # Executar pipeline completo
        try:
            zhang_hybrid.fit_predict()
            
            # Extrair métricas de teste
            test_rmse = zhang_hybrid.metrics_results['test_metrics']['RMSE']
            test_mae = zhang_hybrid.metrics_results['test_metrics']['MAE']
            test_mape = zhang_hybrid.metrics_results['test_metrics']['MAPE']
            
            print(f"  [OK] RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")
            
            results_summary.append({
                'serie': base_name,
                'status': 'OK',
                'rmse': test_rmse,
                'mae': test_mae,
                'mape': test_mape
            })
            
        except Exception as e:
            print(f"  [ERRO] {str(e)}")
            results_summary.append({
                'serie': base_name,
                'status': 'ERRO',
                'erro': str(e)
            })
    
    # Resumo final
    print("\n" + "=" * 80)
    print("RESUMO DA EXECUÇÃO")
    print("=" * 80)
    
    for result in results_summary:
        if result['status'] == 'OK':
            print(f"[✓] {result['serie']:<20} RMSE: {result['rmse']:>8.4f} | MAE: {result['mae']:>8.4f} | MAPE: {result['mape']:>6.2f}%")
        else:
            print(f"[✗] {result['serie']:<20} ERRO: {result['erro']}")
    
    # ========================================================================
    # EXPORTAÇÃO DE RESULTADOS EM CSV
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPORTANDO RESULTADOS")
    print("=" * 80)
    
    # Filtrar apenas resultados bem-sucedidos
    results_ok = [r for r in results_summary if r['status'] == 'OK']
    
    if results_ok:
        # Criar DataFrame
        df = pd.DataFrame(results_ok)
        
        # Padronizar métricas conforme literatura:
        # MSE = RMSE²
        # MAD = MAE (Mean Absolute Deviation)
        df['MSE'] = df['rmse'] ** 2
        df['MAD'] = df['mae']
        
        # Renomear e organizar colunas finais
        df = df[['serie', 'MSE', 'MAD', 'mape']].rename(columns={'mape': 'MAPE'})
        
        # Definir 'serie' como index
        df = df.set_index('serie')
        
        # Imprimir tabela formatada
        print("\nTabela de Resultados - Zhang (2003):")
        print(df.to_string())
        
        # Criar diretório de comparações
        comparisons_dir = os.path.join('data', 'result', 'comparisons')
        os.makedirs(comparisons_dir, exist_ok=True)
        
        # Salvar CSV
        csv_path = os.path.join(comparisons_dir, 'resultados_zhang.csv')
        df.to_csv(csv_path)
        
        print(f"\n[✓] Resultados salvos em: {csv_path}")
    else:
        print("\n[AVISO] Nenhum resultado bem-sucedido para exportar.")
    
    print("=" * 80)
    
    print("\n[CONCLUIDO] Baseline Zhang (2003) executada com sucesso!")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        run_zhang_baseline()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Execução cancelada pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO FATAL] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
