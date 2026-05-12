"""
Script para executar baseline de NoLiC (Santos Júnior et al., 2019).

Roda a arquitetura híbrida NoLiC (Nonlinear Combination) em séries clássicas.

NoLiC possui 3 estágios:
1. Modelo Linear (ARIMA)
2. Modelo Não-Linear nos resíduos (M_NL) com Grid Search
3. Modelo Combinador (M_c) usando CCF para calcular L_max

Referência
----------
Santos Júnior, D. S., et al. (2019). "A hybrid system based on a nonlinear 
combination of ARIMA and neural networks for time series forecasting." 
Applied Soft Computing.
"""

import sys
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
from model.hybrid_system_exp import NoLiCHybrid
import config

# Importar gerador de baselines lineares
from run_arima_base import generate_linear_baselines


# ============================================================================
# CONFIGURAÇÕES POR SÉRIE (test_size exato conforme literatura)
# ============================================================================

SERIES_CONFIG = {
    'sunspot.txt': {
        'test_size': 67,    # Santos Jr. (2019)
        'lag_size': 12,  # Usa PACF automático (framework)
    },
    'lynx.txt': {
        'test_size': 14,    # Santos Jr. (2019)
        'lag_size': 12,  # Usa PACF automático
    },
    'airlines.txt': {
        'test_size': 12,    # Santos Jr. (2024) usou 12 para séries mensais
        'lag_size': 12,  # Usa PACF automático
    }
}


# ============================================================================
# PARÂMETROS GLOBAIS DO EXPERIMENTO
# ============================================================================

# ID do experimento
EXPERIMENT_ID = 'nolic_baseline'

# Nome do modelo ARIMA pré-treinado (será criado se não existir)
LINEAR_MODEL_NAME = '11arima'

# Parâmetros da rede neural base (Grid Search vai testar 1-30 neurônios)
MLP_MAX_ITER = 1000
MLP_RANDOM_STATE = 42

# Validação: OBRIGATÓRIO para Grid Search
# NoLiC exige conjunto de validação para otimização
VAL_SIZE = 15  # 15 observações absolutas

# Horizonte de previsão
HORIZON = 1

# Diferenciação e normalização
DIFF_KPSS = False
NORMALIZE = True


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def run_nolic_baseline():
    """
    Executa baseline de NoLiC (Santos Júnior et al., 2019) nas séries clássicas.
    
    NoLiC = Nonlinear Combination (Combinação Não-Linear)
    - Usa CCF para calcular L_max (atraso máximo significativo)
    - Grid Search independente para M_NL e M_c (1-30 neurônios)
    - Combina previsões lineares e não-lineares de forma otimizada
    """
    series_list = ['sunspot.txt', 'lynx.txt', 'airlines.txt']
    
    print("=" * 80)
    print("BASELINE NOLIC (SANTOS JÚNIOR ET AL., 2019)")
    print("=" * 80)
    print(f"\nExperiment ID: {EXPERIMENT_ID}")
    print(f"Linear Model: {LINEAR_MODEL_NAME}")
    print(f"val_size: {VAL_SIZE} (obrigatório para Grid Search)")
    print(f"Grid Search: 1-30 neurônios para M_NL e M_c")
    print("-" * 80)
    
    # ========================================================================
    # ETAPA 1: GARANTIR EXISTÊNCIA DOS MODELOS ARIMA BASE
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 1/2: GERANDO MODELOS ARIMA BASE")
    print("=" * 80)
    print("Pipeline integrado: Treinando modelos lineares antes dos híbridos...")
    print("NoLiC EXIGE conjunto de validação para Grid Search.")
    print("-" * 80)
    
    arima_results = generate_linear_baselines(
        experiment_id=EXPERIMENT_ID,
        series_config=SERIES_CONFIG,
        val_size=VAL_SIZE,  # IMPORTANTE: validação obrigatória
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
    
    print("\n[✓] Modelos ARIMA base prontos! Iniciando treinamento NoLiC...")
    
    # ========================================================================
    # ETAPA 2: TREINAMENTO DOS MODELOS HÍBRIDOS NOLIC
    # ========================================================================
    print("\n" + "=" * 80)
    print("ETAPA 2/2: TREINANDO MODELOS HÍBRIDOS NOLIC")
    print("=" * 80)
    print("Arquitetura em 3 estágios:")
    print("  1. ARIMA (linear)")
    print("  2. M_NL (rede nos resíduos) com Grid Search")
    print("  3. M_c (combinador) com CCF + Grid Search")
    print("-" * 80)
    
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
        print(f"  lag_size: {lag_size} (automático via PACF)")
        print(f"  val_size: {VAL_SIZE}")
        
        # Configurar parâmetros do experimento
        experiment_params = {
            'linear_model_name': LINEAR_MODEL_NAME,
            'test_size': test_size,
            'val_size': VAL_SIZE,
            'horizon': HORIZON,
            'lag_size': 'auto',
            'diff_kpss': DIFF_KPSS
        }
        
        # Instanciar modelo MLP base (Grid Search vai testar topologias)
        mlp_base = MLPRegressor(
            max_iter=MLP_MAX_ITER,
            random_state=MLP_RANDOM_STATE,
            activation='logistic',
            early_stopping=False  # Desabilitar para Grid Search funcionar bem
        )
        
        # Instanciar experimento NoLiC
        nolic_hybrid = NoLiCHybrid(
            model=mlp_base,
            experiment_id=EXPERIMENT_ID,
            base_name=base_name,
            model_name='nolic_mlp',
            normalize=NORMALIZE,
            experiment_params=experiment_params
        )
        
        # Executar pipeline completo (3 estágios)
        try:
            nolic_hybrid.fit_predict()
            
            # Extrair métricas de teste
            test_rmse = nolic_hybrid.metrics_results['test_metrics']['RMSE']
            test_mae = nolic_hybrid.metrics_results['test_metrics']['MAE']
            test_mape = nolic_hybrid.metrics_results['test_metrics']['MAPE']
            l_max = nolic_hybrid.metrics_results['l_max']
            
            print(f"  [OK] RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")
            print(f"       L_max: {l_max}")
            
            results_summary.append({
                'serie': base_name,
                'status': 'OK',
                'rmse': test_rmse,
                'mae': test_mae,
                'mape': test_mape,
                'l_max': l_max
            })
            
        except Exception as e:
            print(f"  [ERRO] {str(e)}")
            import traceback
            traceback.print_exc()
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
            print(f"[✓] {result['serie']:<20} RMSE: {result['rmse']:>8.4f} | MAE: {result['mae']:>8.4f} | MAPE: {result['mape']:>6.2f}% | L_max: {result['l_max']}")
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
        df = df[['serie', 'MSE', 'MAD', 'mape', 'l_max']].rename(columns={'mape': 'MAPE'})
        
        # Definir 'serie' como index
        df = df.set_index('serie')
        
        # Imprimir tabela formatada
        print("\nTabela de Resultados - NoLiC (Santos Júnior et al., 2019):")
        print(df.to_string())
        
        # Criar diretório de comparações
        comparisons_dir = os.path.join('data', 'result', 'comparisons')
        os.makedirs(comparisons_dir, exist_ok=True)
        
        # Salvar CSV
        csv_path = os.path.join(comparisons_dir, 'resultados_nolic.csv')
        df.to_csv(csv_path)
        
        print(f"\n[✓] Resultados salvos em: {csv_path}")
    else:
        print("\n[AVISO] Nenhum resultado bem-sucedido para exportar.")
    
    print("=" * 80)
    
    print("\n[CONCLUIDO] Baseline NoLiC executado com sucesso!")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    try:
        run_nolic_baseline()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Execução cancelada pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO FATAL] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
