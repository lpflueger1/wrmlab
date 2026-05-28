import os
import pandas as pd
import warnings
import traceback
import hydro_functions_v2 as hf
warnings.filterwarnings('ignore')
root = 'c:/Users/leoni/OneDrive/Documents/Master Umwelting ETH/Semester 2/Labor WRM/Asssignment 2/Assignment 2-20260512/DATA/DATA'
try:
    qD = pd.read_csv(os.path.join(root,'Q_Diepoldsau_m3s.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()
    qG = pd.read_csv(os.path.join(root,'Q_Gisingen_1976-2023.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()
    sscD = pd.read_csv(os.path.join(root,'SSC_Diepoldsau_gL.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()
    sscG = pd.read_csv(os.path.join(root,'SSC_Gisingen_2003-2020.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()
    qD_log = hf.log_transform(qD, 'q_m3s')
    qG_log = hf.log_transform(qG, 'q_m3s')
    sscD_log = hf.log_transform(sscD, sscD.columns[0])
    sscG_log = hf.log_transform(sscG, sscG.columns[0])
    detrended_series = {}
    for station, dfQ, dfC in [('Diepoldsau', qD_log, sscD_log), ('Gisingen', qG_log, sscG_log)]:
        for var, df, col in [('Q', dfQ, 'q_m3s'), ('C', dfC, dfC.columns[0])]:
            key = f'{var}_{station}'
            z,_,_ = hf.seasonal_standardise(df[col].dropna())
            det,_ = hf.test_and_detrend(z.to_frame(name=col), col)
            detrended_series[key] = det.squeeze()
    z_series = {k:v.dropna() for k,v in detrended_series.items()}
    fitted = hf.fit_best_models(z_series)
    print('Fitted keys:', list(fitted.keys()))
    for k,m in fitted.items():
        print(k, type(m).__name__, 'arparams', getattr(m,'arparams',None), 'maparams', getattr(m,'maparams',None))
except Exception:
    traceback.print_exc()
