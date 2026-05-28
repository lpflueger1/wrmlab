from pathlib import Path
path = Path('c:/Users/leoni/OneDrive/Documents/Master Umwelting ETH/Semester 2/Labor WRM/hydro_functions_v2.py')
text = path.read_text(encoding='utf-8')
old = "    results = {\n        'n_points':              len(y),\n        'slope':                 trend.slope,\n        'pvalue':                trend.pvalue,\n        'significant':           significant,\n        'detrend_method':        method,\n        'mean_after_detrend':    detrended.mean(),\n        'variance_after_detrend': np.var(detrended, ddof=1),\n    }"
new = "    results = {\n        'n_points':              len(y),\n        'intercept':             trend.intercept,\n        'slope':                 trend.slope,\n        'pvalue':                trend.pvalue,\n        'significant':           significant,\n        'detrend_method':        method,\n        'mean_after_detrend':    detrended.mean(),\n        'variance_after_detrend': np.var(detrended, ddof=1),\n    }"
if old not in text:
    raise RuntimeError('Pattern not found')
path.write_text(text.replace(old, new), encoding='utf-8')
print('patched hydro_functions_v2.py')
