"""
Script para gerar baselines lineares (ARIMA) para experimentos híbridos.

Este script é modular e reutilizável para diferentes configurações de séries
temporais, permitindo treinar modelos ARIMA base que serão usados como entrada
para arquiteturas híbridas (Zhang, Khashei-Bijari, NoLiC, etc.).

Referências
-----------
- Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis:
  forecasting and control. John Wiley & Sons.
"""

import sys
import numpy as np
from sklearn.base import clone

from input import input
from model.arima import Arima
from model import generics
from metrics import metrics


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def generate_linear_baselines(
    experiment_id,
    series_config,
    val_size=15,
    diff_kpss=False,
    horizon=1,
    force=True,
    model_name='arima'
):
    """
    Gera baselines lineares ARIMA para múltiplas séries temporais.
    
    Parâmetros
    ----------
    experiment_id : str
        Identificador do experimento (usado para organizar arquivos)
    series_config : dict
        Dicionário com configuração por série:
        {
            'serie.txt': {
                'test_size': int,
                'lag_size': int  # usado como seasonal_lag
            },
            ...
        }
    val_size : int, optional (default=15)
        Tamanho do conjunto de validação
    diff_kpss : bool, optional (default=False)
        Se True, aplica diferenciação baseada no teste KPSS
    horizon : int, optional (default=1)
        Horizonte de previsão (1-step-ahead)
    force : bool, optional (default=True)
        Se True, sobrescreve resultados existentes
    model_name : str, optional (default='arima')
        Nome do modelo para identificação nos arquivos
    
    Retorna
    -------
    dict
        Dicionário com resumo das execuções: {serie: {'status': str, 'rmse': float}}
    """
    print("=" * 80)
    print("GERAÇÃO DE BASELINES LINEARES (ARIMA)")
    print("=" * 80)
    print(f"\nExperiment ID: {experiment_id}")
    print(f"Model Name: {model_name}")
    print(f"Val Size: {val_size}")
    print(f"Horizon: {horizon}")
    print(f"Diff KPSS: {diff_kpss}")
    print(f"Force Rerun: {force}")
    print("-" * 80)
    
    results_summary = {}
    
    for base_name, config in series_config.items():
        print(f"\n[INICIO] Processando: {base_name}")
        
        # Extrair configuração da série
        test_size = config['test_size']
        lag_size = config['lag_size']  # usado como seasonal_lag
        
        print(f"  test_size: {test_size}")
        print(f"  seasonal_lag (m): {lag_size}")
        
        try:
            # Formatar nome do arquivo de saída
            fold, title = generics.format_names(
                experiment_id, 
                base_name, 
                f'{horizon}{model_name}'
            )
            
            # Verificar se já existe (skip se force=False)
            if generics.file_exists(title) and (not force):
                print("  [SKIP] Modelo já executado (use force=True para sobrescrever)")
                results_summary[base_name] = {'status': 'SKIP', 'rmse': None}
                continue
            
            # Configuração de execução
            exec_config = {
                "test_size": test_size,
                "val_size": val_size,
                "horizon": horizon,
                "lag_size": 'auto',  # ARIMA não usa windowing
                "diff_kpss": diff_kpss,
                "normalize": False,  # ARIMA trabalha com dados originais
                "type_filter": None
            }
            
            # Carregar e pré-processar dados
            print("  Carregando série temporal...")
            base_info = input.open_format_train_val_test(base_name, exec_config)
            
            (
                ts_univariate,
                df_train,
                df_val,
                df_test,
                min_max_scaler,
                test_size_actual,
                val_size_actual,
                is_stationary,
                original_ts,
                _
            ) = base_info.sequential_return()
            
            # Instanciar modelo ARIMA
            print("  Treinando ARIMA (auto_arima)...")
            model = Arima(seazonal_lag=lag_size, horizon=horizon)
            forecaster = clone(model).set_params(**{})
            
            # Treinar no conjunto de treino (exclui val + test)
            train_end_idx = -(test_size_actual + val_size_actual + horizon - 1)
            if train_end_idx == 0:
                train_end_idx = len(original_ts)
            
            forecaster.fit(original_ts[:train_end_idx])
            
            # Predição walk-forward em val + test
            print("  Gerando previsões walk-forward...")
            train_predict, test_val_predict = forecaster.predict_steps(
                original_ts[train_end_idx:]
            )
            
            # Calcular resíduos (importante para modelos híbridos)
            residual_series = original_ts - np.concatenate([train_predict, test_val_predict])
            
            # Separar val e test
            if val_size_actual > 0:
                val_predict = test_val_predict[:-test_size_actual]
                test_predict = test_val_predict[-test_size_actual:]
            else:
                val_predict = None
                test_predict = test_val_predict[-test_size_actual:]
            
            # Calcular métricas de teste
            test_metrics = metrics.gerenerate_metric_results(
                original_ts[-test_size_actual:], 
                test_predict
            )
            
            # Calcular métricas de validação (se existir)
            if val_size_actual > 0 and val_predict is not None:
                val_metrics = metrics.gerenerate_metric_results(
                    original_ts[-(test_size_actual + val_size_actual):-test_size_actual],
                    val_predict
                )
            else:
                val_metrics = None
            
            # Construir dicionário de resultados
            result_dict = {
                'train_predict': train_predict,
                'val_predict': val_predict,
                'test_predict': test_predict,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'time_exec': {'training': None, 'testing': None},
                'params': forecaster.forecaster.get_params(),
                'best_metric': None,
                'residual_series': residual_series  # Importante para híbridos!
            }
            
            # Salvar resultado
            print("  Salvando resultado...")
            result = generics.ResultExp(result_dict)
            generics.save_result(
                fold, 
                title, 
                [{'experiment': result, 'val_metric': None}]
            )
            
            # Extrair RMSE para resumo
            test_rmse = test_metrics['RMSE']
            test_mae = test_metrics['MAE']
            
            print(f"  [OK] RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
            print(f"  Arquivo salvo: {title}")
            
            results_summary[base_name] = {
                'status': 'OK',
                'rmse': test_rmse,
                'mae': test_mae
            }
            
        except Exception as e:
            print(f"  [ERRO] {str(e)}")
            import traceback
            traceback.print_exc()
            results_summary[base_name] = {
                'status': 'ERRO',
                'erro': str(e)
            }
    
    # Resumo final
    print("\n" + "=" * 80)
    print("RESUMO DA GERAÇÃO DE BASELINES")
    print("=" * 80)
    
    for serie, result in results_summary.items():
        if result['status'] == 'OK':
            print(f"[✓] {serie:<25} RMSE: {result['rmse']:>8.4f} | MAE: {result['mae']:>8.4f}")
        elif result['status'] == 'SKIP':
            print(f"[→] {serie:<25} SKIP (já existe)")
        else:
            print(f"[✗] {serie:<25} ERRO: {result.get('erro', 'Desconhecido')}")
    
    print("\n[CONCLUIDO] Geração de baselines lineares concluída!")
    print("=" * 80)
    
    return results_summary


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Configuração específica para baseline de Khashei & Bijari (2011).
    
    Esta configuração usa os test_size exatos da literatura:
    - sunspot: 67 (Khashei & Bijari, 2011)
    - lynx: 14 (Khashei & Bijari, 2011)
    - airlines: 12 (Santos Jr., 2024)
    """
    
    # Configuração para desbloquear Khashei-Bijari
    EXPERIMENT_ID = 'kb_baseline'
    
    SERIES_CONFIG = {
        'sunspot.txt': {
            'test_size': 67,
            'lag_size': 12  # seasonal_lag (ciclo de ~11 anos)
        },
        'lynx.txt': {
            'test_size': 14,
            'lag_size': 12  # seasonal_lag
        },
        'airlines.txt': {
            'test_size': 12,
            'lag_size': 12  # seasonal_lag (mensal)
        }
    }
    
    VAL_SIZE = 15  # Padrão para experimentos híbridos
    
    print("\n" + "=" * 80)
    print("CONFIGURAÇÃO: BASELINE KHASHEI & BIJARI (2011)")
    print("=" * 80)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"Séries: {list(SERIES_CONFIG.keys())}")
    print(f"Val Size: {VAL_SIZE}")
    print("=" * 80 + "\n")
    
    try:
        results = generate_linear_baselines(
            experiment_id=EXPERIMENT_ID,
            series_config=SERIES_CONFIG,
            val_size=VAL_SIZE,
            diff_kpss=False,
            horizon=1,
            force=True,
            model_name='arima'
        )
        
        # Verificar se todas as séries foram processadas com sucesso
        success_count = sum(1 for r in results.values() if r['status'] == 'OK')
        total_count = len(SERIES_CONFIG)
        
        if success_count == total_count:
            print(f"\n[✓] SUCESSO TOTAL: {success_count}/{total_count} séries processadas")
            sys.exit(0)
        else:
            print(f"\n[!] PARCIAL: {success_count}/{total_count} séries processadas")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n[INTERROMPIDO] Execução cancelada pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO FATAL] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
