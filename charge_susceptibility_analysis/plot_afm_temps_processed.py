import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy as unp, ufloat
from scipy.optimize import curve_fit
import time
from scipy.interpolate import interp1d, griddata

import yaml




# get the U8 data
df8_gs = pd.read_csv('W:\\RunLog\\2025\\U8_afqmc_combined_gs_all.csv')
df8_T0 = pd.read_csv('W:\\RunLog\\2025\\U8_afqmc_combined_finite.csv')
df8_T1 = pd.read_csv('W:\\RunLog\\2025\\U8_dqmc_square_vs_mu_T_all_bonds_avg_seeds.csv')

df8_gs['T'] = 0*df8_gs['n']
df8_gs['ss'] = df8_gs['ss10']*4/3
df8_gs = df8_gs[['n','ns','ss','T']]
df8_T0['T'] = np.round(1/df8_T0['beta'], 3)
df8_T0['ss'] = df8_T0['ss10']*4/3
df8_T0 = df8_T0[['n','ns','ss','T']]

df8_T1['ns'] = df8_T1['Singles']
df8_T1['n'] = df8_T1['Density']
df8_T1['ss'] = (df8_T1['SzSz(0,0,1.0,0.0)']+df8_T1['SxSx(0,0,1.0,0.0)'])/2
df8_T1 = df8_T1[['n','ns','ss','T']]
df8_T1 = df8_T1.loc[(df8_T1['T']>0.25)&(df8_T1['T']<=0.5)]
df8_T1['n'] = 2-df8_T1['n']
df8 = pd.concat([df8_gs, df8_T0, df8_T1])
df8 = df8.sort_values(['T','n'])

# theory curves
# data_int = {'n': [], 'ss': [], 'T': []}
data_int = {'ns': [], 'ss': [], 'T': []}
# data_int = {'n': [], 'ss_norm': [], 'T': []}
for T in sorted(df8['T'].unique()):
    dft = df8.loc[df8['T'] == T]
    # nvals = np.linspace(np.min(dft['n']), np.max(dft['n']), num=100)
    # interpolant = interp1d(dft['n'], dft['ss'], kind='linear')
    # ssvals = interpolant(nvals)
    # data_int['n'] += list(nvals)
    # data_int['ss'] += list(ssvals)
    # data_int['T'] += 100 * [T]

    nsvals = np.linspace(np.min(dft['ns']), np.max(dft['ns']), num=100)
    interpolant = interp1d(dft['ns'], dft['ss'], kind='linear')
    ssvals = interpolant(nsvals)
    data_int['ns'] += list(nsvals)
    data_int['ss'] += list(ssvals)
    data_int['T'] += 100 * [T]

    # nsvals = np.linspace(np.min(dft['n']), np.max(dft['n']), num=100)
    # interpolant = interp1d(dft['n'], dft['ss'].to_numpy()/dft['ns'].to_numpy()**2, kind='linear')
    # ssvals = interpolant(nsvals)
    # data_int['n'] += list(nsvals)
    # data_int['ss_norm'] += list(ssvals)
    # data_int['T'] += 100 * [T]

fig_, ax_ = plt.subplots()

fig, ax = plt.subplots()

df_int = pd.DataFrame(data_int)
cmap=plt.get_cmap('viridis')
for T,dfx in df_int.groupby('T'):
    # ax_.plot(dfx['n'], dfx['ss'], label=f"T/t={T}", color=cmap(T/0.5))
    ax_.plot(dfx['ns'], dfx['ss'], label=f"T/t={T}", color=cmap(T / 0.5))
    # ax_.plot(dfx['n'], dfx['ss_norm'], label=f"T/t={T}", color=cmap(T / 0.5))
ax_.grid()
ax_.legend()
ax_.set_xlabel('Density')
# ax_.set_xlabel('Singles density')


df_exp_cal = pd.read_csv("W:\\RunLog\\2025\\exp_n_to_ns_590.0.csv")
ns_to_n = interp1d(df_exp_cal['ns'], df_exp_cal['n'])

def get_therm(dataset, data_int, ax):
    df = pd.read_csv(f"processed/{dataset}.csv")
    # df = pd.read_csv(f"../../compressibility/new_data_analysis/processed/{dataset}.csv")
    p = df['singles_therm'].to_numpy()[0]
    p_err = df['singles_therm_err'].to_numpy()[0]
    eps = 1e-3
    try:
        density = ns_to_n(p)
        density_err = p_err * (ns_to_n(p + eps) - density) / eps
    except Exception as e:
        print(e)
        return 0,0
    ss = df['corr_therm'].to_numpy()[0]
    ss_err = df['corr_therm_err'].to_numpy()[0]
    ss_norm = ss/p**2
    ss_norm_err = np.sqrt(ss_err**2/p**2 + 4*p_err**2/p**2*ss_norm**2)

    # griddata
    # points = (data_int['n'], data_int['ss'])
    points = (data_int['ns'], data_int['ss'])
    # points = (data_int['n'], data_int['ss_norm'])
    values = data_int['T']
    # xi = (density, ss)
    # temps = griddata(points, values, xi, method='linear').ravel()
    # eps = 1e-3
    # xi = (density + eps, ss)
    # dp = (griddata(points, values, xi, method='linear').ravel() - temps) / eps
    # xi = (density, ss + eps)
    # dss = (griddata(points, values, xi, method='linear').ravel() - temps) / eps
    # terr = ((dp * density_err) ** 2 + (dss * ss_err) ** 2) ** 0.5

    xi = (p, ss)
    temps = griddata(points, values, xi, method='linear').ravel()
    eps = 1e-3
    xi = (p + eps, ss)
    dp = (griddata(points, values, xi, method='linear').ravel() - temps) / eps
    xi = (p, ss + eps)
    dss = (griddata(points, values, xi, method='linear').ravel() - temps) / eps
    terr = ((dp * p_err) ** 2 + (dss * ss_err) ** 2) ** 0.5

    # xi = (density, ss_norm)
    # temps = griddata(points, values, xi, method='linear').ravel()
    # eps = 1e-3
    # xi = (density + eps, ss_norm)
    # dp = (griddata(points, values, xi, method='linear').ravel() - temps) / eps
    # xi = (density, ss_norm + eps)
    # dss = (griddata(points, values, xi, method='linear').ravel() - temps) / eps
    # terr = ((dp * density_err) ** 2 + (dss * ss_norm_err) ** 2) ** 0.5


    # temps = unp.uarray(temps, err)
    print(temps[0], terr[0])
    # ax.errorbar([density], [ss], [ss_err], xerr=[density_err], label=dataset)
    ax.errorbar([p], [ss], [ss_err], xerr=[p_err], label=dataset)
    # ax.errorbar([density], [ss_norm], [ss_norm_err], xerr=[density_err], label=dataset)
    return temps, terr



with open('scan_group_info.yaml', "r") as file:
    scan_groups = yaml.safe_load(file)
# with open('../../compressibility/new_data_analysis/scan_group_info.yaml', "r") as file:
#     scan_groups = yaml.safe_load(file)


for i,dataset in enumerate(scan_groups):
    # if dataset == :
    # if dataset in ['U8_cold_B1_mid_freq1','U8_cold_B1_mid_freq2','U8_cold_A1_mid_freq','U8_cold_A1_hf_med_freq']:
        print('plotting {}'.format(dataset))
        t,te = get_therm(dataset, data_int, ax_)
        ax.errorbar([i], [t], [te], capsize=5, label=dataset)

ax.legend(loc=2, bbox_to_anchor=(1, 1))
ax.grid()
ax.set_ylabel('Temperature')
ax.set_xlabel('Dataset')
ax.set_ylim(0, 0.5)
# plt.tight_layout()

ax_.legend(loc=2, bbox_to_anchor=(1,1))
ax_.set_xlim(0.8, 1)
# ax_.set_xlim(0.7, 0.9)
ax_.set_ylabel('Spin correlation')
plt.tight_layout()
plt.show()