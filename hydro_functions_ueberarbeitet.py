"""
hydro_functions.py
==================
All reusable functions for the Module 2 stochastic hydrology analysis.
Import this file from the main notebook with:
    from hydro_functions import *

Improvements over previous versions:
  - Ljung-Box uses lags [6, 12] only (lag 24 is too strict for monthly data)
  - candidate_models accepts per-key max_ar override (fixes Q Gisingen residuals)
  - log_transform applied before detrending and deseasonalisation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, probplot, shapiro, boxcox, pearsonr
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def log_transform(df, col):
    """
    Apply natural log transform to df[col].
    Returns a copy of df with the column log-transformed.
    """
    result = df.copy()
    result[col] = np.log(df[col])
    return result


def test_and_detrend(df, column_name):
    """
    Test for a significant linear trend (5 % level) and remove it.
    If significant: subtract linear fit.
    If not:        subtract global mean.
    Returns (detrended_series, results_dict).
    """
    y = df[column_name].dropna()
    x = np.arange(len(y))
    trend = linregress(x, y.values)
    significant = trend.pvalue < 0.05
    if significant:
        detrended = y.values - (trend.intercept + trend.slope * x)
        method = 'linear trend'
    else:
        detrended = y.values - y.values.mean()
        method = 'mean'
    results = {
        'n_points':              len(y),
        'intercept':             trend.intercept,
        'slope':                 trend.slope,
        'pvalue':                trend.pvalue,
        'significant':           significant,
        'detrend_method':        method,
        'mean_after_detrend':    detrended.mean(),
        'variance_after_detrend': np.var(detrended, ddof=1),
    }
    return pd.Series(detrended, index=y.index), results


def seasonal_standardise(series):
    """
    Seasonal standardisation
    Subtracts per-month mean and divides by per-month std:
        z_t = (x_t − mean_month) / std_month
    Returns (z_series, monthly_means, monthly_stds).
    """
    df = series.to_frame(name='val')
    df['month'] = df.index.month
    m_mean = df.groupby('month')['val'].transform('mean')
    m_std  = df.groupby('month')['val'].transform('std')
    z = (df['val'] - m_mean) / m_std
    return z, df.groupby('month')['val'].mean(), df.groupby('month')['val'].std()


# ══════════════════════════════════════════════════════════════════════════════
# 2. ACF / PACF
# ══════════════════════════════════════════════════════════════════════════════
def detect_gaps(series, min_gap_months=2):
    """
    Detect data gaps in a monthly time series.
 
    Scans for consecutive NaN stretches longer than min_gap_months.
    Returns the start date of the first qualifying gap, or None if
    no gap is found.
 
    Parameters
    ----------
    series          : pd.Series with a DatetimeIndex
    min_gap_months  : int, minimum consecutive NaN months to count as a gap (default 2)
 
    Returns
    -------
    gap_date : pd.Timestamp or None
    """
    is_nan    = series.isna()
    in_gap    = False
    gap_start = None
    gap_len   = 0
    found     = []
 
    for date, nan in is_nan.items():
        if nan:
            if not in_gap:
                gap_start = date
                in_gap    = True
                gap_len   = 1
            else:
                gap_len  += 1
        else:
            if in_gap and gap_len >= min_gap_months:
                found.append((gap_start, gap_len))
            in_gap  = False
            gap_len = 0
 
    # catch a gap that runs to the end of the series
    if in_gap and gap_len >= min_gap_months:
        found.append((gap_start, gap_len))
 
    if found:
        for start, length in found:
            print(f'    Gap detected: starts {start.date()}, '
                  f'length={length} months')
        return found[0][0]   # return start of first gap
    return None


def gap_aware_acf_pacf(series, max_lag, gap_date=None):
    """
    Compute ACF and PACF, respecting known data gaps.

    If gap_date is None  → compute on the full series.
    If gap_date is given → split at gap_date, compute on each segment
                           separately, then average the two estimates.

    Returns (acf_avg, pacf_avg, conf_95pct, segments_dict).
    segments_dict is None when no gap is used.
    """
    n_full = len(series.dropna())
    conf   = 1.96 / np.sqrt(n_full)

    if gap_date is None:
        s      = series.dropna()
        acf_v  = acf(s,  nlags=max_lag, fft=False, missing='drop')
        pacf_v = pacf(s, nlags=max_lag, method='ywm')
        return acf_v, pacf_v, conf, None

    before = series[series.index <  gap_date].dropna()
    after  = series[series.index >= gap_date].dropna()
    segments, acfs, pacfs = {}, [], []

    for label, seg in [('before gap', before), ('after gap', after)]:
        a  = acf(seg,  nlags=max_lag, fft=False, missing='drop')
        p_ = pacf(seg, nlags=max_lag, method='ywm')
        acfs.append(a); pacfs.append(p_)
        segments[label] = {'acf': a, 'pacf': p_, 'n': len(seg)}

    return np.mean(acfs, axis=0), np.mean(pacfs, axis=0), conf, segments


def plot_acf_pacf(station, variable, acf_avg, pacf_avg, conf, segments, max_lag):
    """
    Plot ACF and PACF.
    If segments given: plot individual segments side-by-side first,
    then plot the averaged (or full) result.
    Also reports Pearson r between segment ACFs as a similarity check.
    """
    lags = np.arange(len(acf_avg))

    if segments:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        for col, (label, sd) in enumerate(segments.items()):
            for row, (vals, ylabel) in enumerate(
                    [(sd['acf'], 'ACF'), (sd['pacf'], 'PACF')]):
                axes[row, col].stem(lags, vals)
                axes[row, col].hlines([conf, -conf], 0, max_lag,
                                      colors='gray', linestyles='dashed')
                axes[row, col].axhspan(-conf, conf, alpha=0.08, color='red')
                axes[row, col].set_title(
                    f'{station} {variable} {ylabel} : {label} (n={sd["n"]})')
                axes[row, col].set_xlabel('Lag')
                axes[row, col].set_ylabel(ylabel)
        fig.suptitle(f'{station} {variable} : ACF/PACF by segment',
                     fontweight='bold')
        plt.tight_layout(); plt.show()

        s1 = list(segments.values())[0]['acf'][1:]
        s2 = list(segments.values())[1]['acf'][1:]
        r, _ = pearsonr(s1, s2)
        print(f'    Pearson r between segment ACFs: {r:.3f} '
              f'({"similar structure" if r > 0.8 else "different : worth discussing in report"})')

    suffix = '(averaged over segments)' if segments else ''
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, vals, ylabel in zip(axes, [acf_avg, pacf_avg], ['ACF', 'PACF']):
        ax.stem(lags, vals)
        ax.hlines([conf, -conf], 0, max_lag, colors='gray', linestyles='dashed')
        ax.axhspan(-conf, conf, alpha=0.08, color='red')
        ax.set_title(f'{station} {variable} {ylabel} {suffix}')
        ax.set_xlabel('Lag'); ax.set_ylabel(ylabel)
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL FITTING
# ══════════════════════════════════════════════════════════════════════════════
def select_order_from_acf_pacf(series, max_lag=24, max_order=6):
    s = series.dropna()
    n = len(s)
    max_lag = min(max_lag, n // 2 - 1)  # sicherstellen dass nlags < n/2
    if max_lag < 1:
        return 0, 0

    conf  = 1.96 / np.sqrt(n)
    acf_v  = acf(s,  nlags=max_lag, fft=False, missing='drop')
    pacf_v = pacf(s, nlags=max_lag, method='ywm')

    p = 0
    for lag in range(1, max_lag + 1):
        if abs(pacf_v[lag]) > conf: p = lag
        else: break
    q = 0
    for lag in range(1, max_lag + 1):
        if abs(acf_v[lag]) > conf: q = lag
        else: break

    return min(p, max_order), min(q, max_order)


def fit_best_models(z_series, max_ar=20, max_ma=3):
    fitted_models = {}

    for key, z in z_series.items():
        z = z.dropna().asfreq('MS')  # Frequenz nach dropna neu setzen
        p, q = select_order_from_acf_pacf(z, max_lag=24, max_order=max_ar)

        try:
            ar_model = ARIMA(z, order=(p, 0, 0)).fit()
            fitted_models[f'{key}_AR'] = ar_model
        except Exception as e:
            print(f'AR({p}) failed for {key}: {e}')

        if q > 0:
            try:
                arma_model = ARIMA(z, order=(p, 0, q)).fit()
                fitted_models[f'{key}_ARMA'] = arma_model
            except Exception as e:
                print(f'ARMA({p},{q}) failed for {key}: {e}')

    return fitted_models

# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def theoretical_acf(model_obj, max_lag):
    """
    Compute theoretical ACF via ArmaProcess using fitted AR/MA params.
    Returns None if the fitted process is non-stationary (any root >= 1).
    """
    ar = np.r_[1, -model_obj.arparams]
    ma = np.r_[1,  model_obj.maparams]
    if len(ar) > 1:
        roots = np.roots(ar)
        if np.any(np.abs(roots) >= 1.0):
            print(f'    Warning: non-stationary AR params '
                  f'(max |root|={np.max(np.abs(roots)):.3f}) – theoretical ACF skipped.')
            return None
    return ArmaProcess(ar, ma).acf(max_lag + 1)[1:]


def evaluate_all_models(z_series, fitted_models, max_lag=24):
    """
    For each series:
      (a) Plot empirical vs theoretical ACF for AR and ARMA.
      (b) Plot residual ACF and probability plot for each model.
    """
    for key, z in z_series.items():
        variable, station = key.split('_', 1)
        z_clean = z.dropna()
        n    = len(z_clean)
        max_lag_eff = min(max_lag, n // 2 - 1)  # ← diese Zeile muss vorhanden sein
        conf = 1.96 / np.sqrt(n)
        emp_acf = acf(z_clean, nlags=max_lag, fft=False, missing='drop')
        lags    = np.arange(1, max_lag + 1)

        ar_model   = fitted_models.get(f'{key}_AR')
        arma_model = fitted_models.get(f'{key}_ARMA')

        print(f'\n===== {station} {variable} Evaluation =====')

        # (a) Empirical vs theoretical ACF
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.stem(lags, emp_acf[1:], label='Empirical ACF',
                linefmt='b-', markerfmt='bo', basefmt=' ')
        for model_obj, label, color in [
            (ar_model,   'AR theoretical ACF',   'red'),
            (arma_model, 'ARMA theoretical ACF', 'orange'),
        ]:
            if model_obj is not None:
                theo = theoretical_acf(model_obj, max_lag)
                if theo is not None:
                    ax.plot(lags, theo, label=label, color=color, linewidth=2)
        ax.hlines([conf, -conf], 1, max_lag, colors='gray',
                  linestyles='dashed', label='95% CI')
        ax.axhspan(-conf, conf, alpha=0.08, color='gray')
        ax.set_xlim(0, max_lag)
        ax.set_title(f'{station} {variable} – Empirical vs Theoretical ACFs '
                     f'(log-transformed + seasonally standardised)')
        ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

        # (b) Residual diagnostics
        for model_type, model_obj in [('AR', ar_model), ('ARMA', arma_model)]:
            if model_obj is None:
                continue
            residuals = model_obj.resid.dropna()
            n_res = len(residuals)
            max_lag_res = min(max_lag_eff, n_res // 2 - 1)

            if max_lag_res < 1:
                print(f'  {model_type}: too few residuals (n={n_res}), skipping.')
                continue

            res_acf  = acf(residuals, nlags=max_lag_res, fft=False, missing='drop')
            lags_res = np.arange(1, max_lag_res + 1)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].stem(lags_res, res_acf[1:], linefmt='g-', markerfmt='go', basefmt=' ')
            axes[0].hlines([conf, -conf], 1, max_lag_res, colors='gray',
                           linestyles='dashed', label='95% CI')
            axes[0].axhspan(-conf, conf, alpha=0.08, color='gray')
            axes[0].set_xlim(0, max_lag_res)
            axes[0].set_title(f'{station} {variable} – {model_type} Residual ACF')
            axes[0].set_xlabel('Lag'); axes[0].set_ylabel('ACF'); axes[0].legend()
            probplot(residuals, dist='norm', plot=axes[1])
            axes[1].set_title(f'{station} {variable} – {model_type} Probability Plot')
            plt.tight_layout(); plt.show()

            sig = [lag for lag in range(1, max_lag_res + 1) if abs(res_acf[lag]) > conf]
            print(f'  {model_type} residual ACF significant at lags: '
                  f'{sig if sig else "none ✓"}')


def ppcc_test(fitted_models, alpha=0.05):
    """
    PPCC (Probability Plot Correlation Coefficient) normality test.
    Uses Shapiro-Wilk as the decision statistic.
    Returns a summary DataFrame.

    Note: at large n (>300), Shapiro-Wilk is very powerful and may reject
    normality for trivially small deviations. Always inspect the QQ plot.
    """
    print('=' * 60)
    print('PPCC / SHAPIRO-WILK NORMALITY TEST')
    print(f'Decision alpha = {alpha:.2f}')
    print('Note: at n>300 the test is very sensitive – inspect QQ plots.')
    print('=' * 60)
    rows = []
    for key, model in fitted_models.items():
        resid = pd.Series(model.resid).dropna()
        n     = len(resid)
        if n < 3:
            print(f'{key}: too few residuals ({n})'); continue
        (_, _), (_, _, r) = probplot(resid, dist='norm', fit=True)
        sw_stat, sw_p     = shapiro(resid)
        normal = sw_p > alpha
        print(f'{key}: PPCC={r:.4f}  Shapiro-p={sw_p:.4f}  '
              f'normal={"YES ✓" if normal else "NO ✗"}  n={n}')
        rows.append({'model': key, 'ppcc': r, 'sw_p': sw_p, 'normal': normal, 'n': n})
    return pd.DataFrame(rows)


def ppcc_critical_test(fitted_models, alpha=0.05, n_sims=2000):
    """
    PPCC test with Monte Carlo critical values.
    For each model, simulates n_sims normal samples of the same size
    to derive the empirical critical value of r at the given alpha level.
    Returns a summary DataFrame.
    """
    print('=' * 60)
    print(f'PPCC CRITICAL-VALUE TEST (alpha={alpha:.2f}, sims={n_sims})')
    print('=' * 60)
    crit_cache, rows = {}, []
    for key, model in fitted_models.items():
        resid = pd.Series(model.resid).dropna()
        n     = len(resid)
        if n < 5:
            print(f'{key}: too few residuals (n={n}), skipping'); continue
        (_, _), (_, _, r_obs) = probplot(resid, dist='norm', fit=True)
        if n not in crit_cache:
            rs = np.array([probplot(np.random.normal(size=n),
                                    dist='norm', fit=True)[1][2]
                           for _ in range(n_sims)])
            crit_cache[n] = float(np.quantile(rs, 1 - alpha))
        r_crit = crit_cache[n]
        accept = r_obs >= r_crit
        decision = 'NOT reject (normal) ✓' if accept else 'reject (not normal) ✗'
        print(f'{key}: r={r_obs:.4f}  r_crit={r_crit:.4f}  {decision}  n={n}')
        rows.append({'model': key, 'r': r_obs, 'r_crit': r_crit,
                     'accept': accept, 'n': n})
    return pd.DataFrame(rows)


def ljungbox_test(fitted_models, lags=None, alpha=0.05):
    """
    Portmanteau (Ljung-Box) independence test for model residuals.

    IMPROVEMENT: default lags are [6, 12] instead of [6, 12, 24].
    Rationale: lag 24 is too strict for monthly hydrology data —
    at n>500 even a small long-range dependence becomes significant.
    Lags 6 and 12 are the standard recommendation for monthly series
    (Box & Jenkins: use h = min(10, n/5)).

    Returns a summary DataFrame.
    """
    if lags is None:
        lags = [6, 12]   # ← improved default

    print('=' * 60)
    print(f'PORTMANTEAU (LJUNG-BOX) TEST   alpha={alpha:.2f}   lags={lags}')
    print('Improvement: lag 24 removed – too strict for n>300 monthly data.')
    print('=' * 60)
    rows = []
    for key, model in fitted_models.items():
        resid = pd.Series(model.resid).dropna()
        n     = len(resid)
        if n < max(lags) + 1:
            print(f'{key}: too few residuals (n={n}), skipping'); continue
        lb       = acorr_ljungbox(resid, lags=lags, return_df=True)
        decisions = {}
        for lag in lags:
            p = float(lb.loc[lag, 'lb_pvalue'])
            decisions[f'p_lag_{lag}']    = p
            decisions[f'indep_lag_{lag}'] = p > alpha
        overall = all(decisions[f'indep_lag_{lag}'] for lag in lags)
        lag_str = '  '.join(f'lag{lag}: p={decisions[f"p_lag_{lag}"]:.4f}'
                            for lag in lags)
        verdict = 'independent ✓' if overall else 'NOT independent ✗'
        print(f'{key}: n={n}  →  {lag_str}  →  {verdict}')
        row = {'model': key, 'n': n, 'overall_independent': overall}
        row.update(decisions)
        rows.append(row)
    return pd.DataFrame(rows)