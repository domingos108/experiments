"""
nonlinearity_test.py
--------------------
Teste estatístico de não-linearidade remanescente em resíduos de modelos
lineares usando o teste BDS (Brock-Dechert-Scheinkman).

Uso típico
----------
Após ajustar um modelo ARIMA, calcule os resíduos e teste se contêm
estrutura não-linear que justifique o uso de um modelo híbrido:

    >>> import pandas as pd
    >>> from nonlinearity_test import test_nonlinearity
    >>> 
    >>> # residuals = y_true - y_pred_arima
    >>> residuals = pd.Series([...])
    >>> 
    >>> if test_nonlinearity(residuals):
    ...     print("Não-linearidade detectada → usar híbrido ARIMA-ML")
    ... else:
    ...     print("Resíduos são ruído branco → usar apenas ARIMA")

Teoria
------
O teste BDS (Brock, Dechert, Scheinkman, 1996) detecta dependência temporal
não-linear em séries temporais. É um teste de hipótese:

    H0: os resíduos são i.i.d. (independentes e identicamente distribuídos)
    H1: existe estrutura de dependência (linear ou não-linear) remanescente

Se rejeitamos H0 (p-value < threshold), há evidência de que os resíduos
não são ruído branco puro, sugerindo que um modelo ML pode capturar padrões
adicionais.

Referência
----------
Brock, W. A., Dechert, W., Scheinkman, J., & LeBaron, B. (1996).
"A test for independence based on the correlation dimension."
Econometric Reviews, 15(3), 197-235.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def test_nonlinearity(
    residuals: pd.Series | np.ndarray,
    p_value_threshold: float = 0.05,
    max_dim: int = 3,
    epsilon: Optional[float] = None,
) -> bool:
    """
    Testa se há não-linearidade remanescente em resíduos usando o teste BDS.

    Parâmetros
    ----------
    residuals : pd.Series | np.ndarray
        Resíduos de um modelo linear (ex: y_true - y_pred_arima).
        Devem ter pelo menos 100 observações para estabilidade estatística.

    p_value_threshold : float, padrão=0.05
        Nível de significância. Se p-value < threshold, rejeitamos H0
        (i.i.d.) e concluímos que há não-linearidade.

    max_dim : int, padrão=3
        Dimensão máxima de embedding para o teste BDS. Valores típicos: 2-5.
        Dimensões maiores detectam padrões mais complexos, mas exigem mais dados.

    epsilon : float | None, padrão=None
        Raio de proximidade para o teste (em unidades de desvio padrão dos resíduos).
        Se None, usa 1.5 × std(residuals) como padrão conservador.

    Retorna
    -------
    bool
        True  → há evidência de não-linearidade (usar modelo híbrido)
        False → resíduos parecem ruído branco (usar apenas ARIMA)

    Raises
    ------
    ValueError
        Se os resíduos tiverem menos de 100 observações ou variância zero.

    Notas
    -----
    - O teste BDS é sensível ao tamanho da amostra. Com N < 100, a potência
      do teste é limitada e pode não detectar não-linearidade sutil.
    - Se o teste falhar por convergência numérica, assume-se conservadoramente
      que NÃO há não-linearidade (retorna False), evitando overfitting.

    Exemplos
    --------
    >>> residuals = pd.Series(np.random.randn(200))  # ruído branco simulado
    >>> test_nonlinearity(residuals)
    False

    >>> # Série com estrutura não-linear (quadrática)
    >>> x = np.linspace(0, 10, 200)
    >>> y = x**2 + np.random.randn(200) * 0.5
    >>> residuals = pd.Series(y - np.polyfit(x, y, 1) @ [x, np.ones_like(x)])
    >>> test_nonlinearity(residuals)
    True
    """
    # --- Validações ---
    if isinstance(residuals, pd.Series):
        residuals = residuals.values

    residuals = np.asarray(residuals).flatten()

    if len(residuals) < 100:
        warnings.warn(
            f"BDS test tem baixa potência com N={len(residuals)} < 100. "
            "Retornando False (sem não-linearidade detectada) por segurança.",
            UserWarning,
        )
        return False

    if np.var(residuals) == 0:
        raise ValueError(
            "Resíduos têm variância zero (constantes). "
            "Não é possível executar o teste BDS."
        )

    # Remove NaN/Inf
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 100:
        warnings.warn(
            "Após remoção de NaN/Inf, restam menos de 100 observações válidas. "
            "Retornando False por segurança.",
            UserWarning,
        )
        return False

    # --- Configuração do teste ---
    if epsilon is None:
        # Padrão conservador: 1.5 × std (BDS recomenda 0.5 a 2.0)
        epsilon = 1.5 * np.std(residuals, ddof=1)

    # --- Execução do teste BDS ---
    try:
        from statsmodels.tsa.stattools import bds

        # bds() retorna (statistic, p_value)
        # Testamos múltiplas dimensões de embedding (2, 3, ..., max_dim)
        # e usamos a MENOR p-value encontrada (teste mais conservador)
        p_values = []

        for dim in range(2, max_dim + 1):
            try:
                _, p_val = bds(residuals, max_dim=dim, epsilon=epsilon, distance=1.5)
                # bds pode retornar escalar (dim=1) ou array (dim > 1)
                # Garantimos conversão para array e pegamos o último valor
                p_val = np.atleast_1d(p_val)
                p_values.append(float(p_val[-1]))
            except Exception as e:
                # Falha em uma dimensão específica — continua para as outras
                warnings.warn(
                    f"BDS falhou para dimensão {dim}: {e}. Ignorando.",
                    RuntimeWarning,
                )
                continue

        if not p_values:
            # Todas as dimensões falharam
            warnings.warn(
                "BDS test falhou em todas as dimensões testadas. "
                "Assumindo ausência de não-linearidade (retorno conservador: False).",
                RuntimeWarning,
            )
            return False

        # Menor p-value entre todas as dimensões
        min_p_value = min(p_values)

        # Decisão: se p-value < threshold → rejeita H0 → há não-linearidade
        return min_p_value < p_value_threshold

    except ImportError:
        raise ImportError(
            "statsmodels não está instalado. Execute: pip install statsmodels"
        )

    except Exception as e:
        # Erro inesperado no teste BDS (convergência numérica, etc.)
        warnings.warn(
            f"BDS test falhou com erro inesperado: {e}. "
            "Retornando False (sem não-linearidade) por segurança.",
            RuntimeWarning,
        )
        return False


def test_nonlinearity_report(
    residuals: pd.Series | np.ndarray,
    p_value_threshold: float = 0.05,
    max_dim: int = 3,
    epsilon: Optional[float] = None,
) -> dict:
    """
    Versão estendida de test_nonlinearity() que retorna um relatório detalhado.

    Retorna
    -------
    dict
        {
            "has_nonlinearity": bool,
            "p_values": list[float],       # p-value para cada dimensão
            "min_p_value": float,
            "epsilon_used": float,
            "n_obs": int,
            "decision": str,
        }

    Exemplos
    --------
    >>> report = test_nonlinearity_report(residuals)
    >>> print(report["decision"])
    "Não-linearidade detectada (min p-value=0.003 < 0.05)"
    """
    residuals_clean = np.asarray(residuals).flatten()
    residuals_clean = residuals_clean[np.isfinite(residuals_clean)]
    n_obs = len(residuals_clean)

    if epsilon is None:
        epsilon = 1.5 * np.std(residuals_clean, ddof=1) if n_obs > 1 else 0.0

    if n_obs < 100:
        return {
            "has_nonlinearity": False,
            "p_values": [],
            "min_p_value": np.nan,
            "epsilon_used": epsilon,
            "n_obs": n_obs,
            "decision": f"Amostra insuficiente (N={n_obs} < 100). Decisão: não usar ML.",
        }

    try:
        from statsmodels.tsa.stattools import bds

        p_values = []
        for dim in range(2, max_dim + 1):
            try:
                _, p_val = bds(residuals_clean, max_dim=dim, epsilon=epsilon, distance=1.5)
                p_val = np.atleast_1d(p_val)
                p_values.append(float(p_val[-1]))
            except Exception:
                continue

        if not p_values:
            return {
                "has_nonlinearity": False,
                "p_values": [],
                "min_p_value": np.nan,
                "epsilon_used": epsilon,
                "n_obs": n_obs,
                "decision": "Teste BDS falhou em todas as dimensões. Decisão conservadora: não usar ML.",
            }

        min_p = min(p_values)
        has_nl = min_p < p_value_threshold

        decision = (
            f"Não-linearidade detectada (min p-value={min_p:.4f} < {p_value_threshold})"
            if has_nl
            else f"Resíduos são i.i.d. (min p-value={min_p:.4f} >= {p_value_threshold})"
        )

        return {
            "has_nonlinearity": has_nl,
            "p_values": p_values,
            "min_p_value": min_p,
            "epsilon_used": epsilon,
            "n_obs": n_obs,
            "decision": decision,
        }

    except Exception as e:
        return {
            "has_nonlinearity": False,
            "p_values": [],
            "min_p_value": np.nan,
            "epsilon_used": epsilon,
            "n_obs": n_obs,
            "decision": f"Erro inesperado: {e}. Decisão conservadora: não usar ML.",
        }


# ---------------------------------------------------------------------------
# Exemplo de uso standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Teste 1: Ruído branco puro (deve retornar False)
    print("Teste 1: Ruído branco simulado (N=200)")
    white_noise = pd.Series(np.random.randn(200))
    result = test_nonlinearity(white_noise)
    print(f"  → Não-linearidade detectada: {result}")
    print(f"  → Decisão: {'Usar híbrido' if result else 'Usar apenas ARIMA'}\n")

    # Teste 2: Série com padrão quadrático remanescente (deve retornar True)
    print("Teste 2: Série com não-linearidade quadrática")
    np.random.seed(42)
    x = np.linspace(0, 10, 200)
    y_nonlinear = 0.5 * x**2 + np.random.randn(200) * 2
    # Ajuste linear inadequado → resíduos contêm padrão quadrático
    linear_fit = np.polyval(np.polyfit(x, y_nonlinear, 1), x)
    residuals_nl = pd.Series(y_nonlinear - linear_fit)
    result_nl = test_nonlinearity(residuals_nl)
    print(f"  → Não-linearidade detectada: {result_nl}")
    print(f"  → Decisão: {'Usar híbrido' if result_nl else 'Usar apenas ARIMA'}\n")

    # Teste 3: Relatório detalhado
    print("Teste 3: Relatório detalhado do teste 2")
    report = test_nonlinearity_report(residuals_nl)
    for key, val in report.items():
        print(f"  {key:20s}: {val}")
