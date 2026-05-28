import os
import pandas as pd
import matplotlib.pyplot as plt

root = r'c:\Users\leoni\OneDrive\Documents\Master Umwelting ETH\Semester 2\Labor WRM\Asssignment 2\Assignment 2-20260512\DATA\DATA'
q1 = pd.read_csv(os.path.join(root,'Q_Diepoldsau_m3s.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()
q2 = pd.read_csv(os.path.join(root,'Q_Gisingen_1976-2023.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()
ssc1 = pd.read_csv(os.path.join(root,'SSC_Diepoldsau_gL.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()
ssc2 = pd.read_csv(os.path.join(root,'SSC_Gisingen_2003-2020.csv'), parse_dates=['timestamp']).set_index('timestamp').resample('ME').mean()

for station, dfQ, dfC in [('Diepoldsau', q1, ssc1), ('Gisingen', q2, ssc2)]:
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(dfQ.index, dfQ['q_m3s'], label='Q (m3/s)', color='tab:blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Q (m3/s)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(dfC.index, dfC.iloc[:,0], label='C (g/L)', color='tab:red', alpha=0.7)
    ax2.set_ylabel('C (g/L)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    fig.suptitle(f'{station} monthly mean Q and C')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
