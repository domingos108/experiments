"""
Script para executar baseline de Khashei & Bijari (2011).

Roda a arquitetura híbrida não-linear em séries clássicas usadas no paper original.

Referência
----------
Khashei, M., & Bijari, M. (2011). "A novel hybridization of artificial
neural networks and ARIMA models for time series forecasting."
Applied Soft Computing, 11(2), 2664-2675.
"""

import sys
from sklearn.neural_network import MLPRegressor
from model.hybrid_system_exp import KhasheiBijariHybrid
import config

# Importar gerador de baselines lineares
from run_arima_base import generate_linear_baselines


# ============================================================================
# CONFIGURAÇÕES POR SÉRIE (test_size exato conforme literatura)
# ============================================================================

SERIES_CONFIG = {
    'sunspot.txt': {
        'test_size': 67,    # Khashei & Bijari (2011)
        'lag_size': 12,      # Padrão para séries anuais com ciclo de 11 anos
    },
    'lynx.txt': {
        'test_size': 14,    # Khashei & Bijari (2011)
        'lag_size': 12,      # Padrão para séries anuais
    },
    'airlines.txt': {
        'test_size': 12,    # Santos Jr. (2024) usou 12 para séries mensais
        'lag_size': 12,      # Padrão para séries mensais (1 ano)
    }
}


# ============================================================================
# PARÂMETROS GLOBAIS DO EXPERIMENTO
# ============================================================================

# ID do experimento (usado para organização dos resultados)
EXPERIMENT_ID = 'kb_baseline'

# Nome do modelo ARIMA pré-treinado (deve existir antes de rodar este script)
LINEAR_MODEL_NAME = '1arima'

# Arquitetura da rede neural (topologia parcimoniosa inicial)
# Khashei não especificou topologia exata, usar arquitetura simples
MLP_HIDDEN_LAYERS = (5,)
MLP_MAX_ITER = 1000
MLP_RANDOM_STATE = 42

# Parâmetros Khashei & Bijari
N1_LAGS = 3  # Lags dos resíduos ARIMA
M1_LAGS = 3  # Lags da série original

# Validação: autores não usaram conjunto de validação explícito
VAL_SIZE = 0

# Horizonte de previsão
HORIZON = 1

# Diferenciação e normalização
DIFF_KPSS = False
NORMALIZE = True


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def run_khashei_bijari_baseline():
    """
    Executa baseline de Khashei & Bijari (2011) nas séries clássicas.
    """
    series_list = ['sunspot.txt', 'lynx.txt', 'airlines.txt']
    
    print("=" * 80)
    print("BASELINE KHASHEI & BIJARI (2011)")
    print("=" * 80)
    print(f"\nExperiment ID: {EXPERIMENT_ID}")
    print(f"Linear Model: {LINEAR_MODEL_NAME}")
    print(f"MLP Topology: {MLP_HIDDEN_LAYERS}")
    print(f"n1_lags: {N1_LAGS} | m1_lags: {M1_LAGS}")
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
    # ETAPA 2: TREINAMENTO DOS MODELOS HÍBRIDOS KHASHEI-BIJARI
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 2/2: TREINANDO MODELOS HÍBRIDOS KHASHEI & BIJARI")
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
        
        print(f"  test_size: {test_size}")
        print(f"  lag_size: {lag_size}")
        
        # Configurar parâmetros do experimento
        experiment_params = {
            'linear_model_name': LINEAR_MODEL_NAME,
            'n1_lags': N1_LAGS,
            'm1_lags': M1_LAGS,
            'test_size': test_size,
            'val_size': VAL_SIZE,
            'horizon': HORIZON,
            'lag_size': lag_size,
            'diff_kpss': DIFF_KPSS
        }
        
        # Instanciar modelo MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            max_iter=MLP_MAX_ITER,
            random_state=MLP_RANDOM_STATE,
            #activation='relu',
            activation='logistic', 
            solver='adam'
        )
        
        # Instanciar experimento Khashei-Bijari
        kb_hybrid = KhasheiBijariHybrid(
            model=mlp,
            experiment_id=EXPERIMENT_ID,
            base_name=base_name,
            model_name='kb_mlp',
            normalize=NORMALIZE,
            experiment_params=experiment_params
        )
        
        # Executar pipeline completo
        try:
            kb_hybrid.fit_predict()
            
            # Extrair métricas de teste
            test_rmse = kb_hybrid.metrics_results['test_metrics']['RMSE']
            test_mae = kb_hybrid.metrics_results['test_metrics']['MAE']
            test_mape = kb_hybrid.metrics_results['test_metrics']['MAPE']
            
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
    
    print("\n[CONCLUIDO] Baseline Khashei & Bijari (2011) executada com sucesso!")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        run_khashei_bijari_baseline()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Execução cancelada pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO FATAL] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
