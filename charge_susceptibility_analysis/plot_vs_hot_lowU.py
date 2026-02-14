import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy as unp, ufloat
from scipy.optimize import curve_fit
import time
import os
from scipy.interpolate import interp1d, griddata

from matplotlib.colors import to_rgb

import yaml

import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.labelpad'] = 0
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['axes.titlepad'] = 0
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['ytick.major.pad'] = 1
mpl.rcParams['ytick.minor.pad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['xtick.minor.pad'] = 1
mpl.rcParams['lines.markeredgewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 4.5




fig_, ax_ = plt.subplots(nrows=3, sharex=True, figsize=(6, 8))


# df_exp_cal = pd.read_csv("W:\\RunLog\\2025\\exp_n_to_ns_590.0.csv")
# df_exp_cal = pd.read_csv(os.path.join('/Volumes','share','FileServer','RunLog','2025','exp_n_to_ns_590.0.csv'))
df_exp_cal = pd.read_csv(os.path.join('aux','exp_n_to_ns_590.0.csv'))
ns_to_n = interp1d(df_exp_cal['ns'], df_exp_cal['n'])

def get_vals(dataset, ax_, is_compensated):
    df = pd.read_csv(f"processed/{dataset}.csv").sort_values('periods')
    p = df['avg_singles'].to_numpy()[0]
    p_err = df['avg_singles_err'].to_numpy()[0]
    eps=1e-3
    try:
        density = ns_to_n(p)
        density_err = p_err * (ns_to_n(p + eps) - density) / eps
    except Exception as e:
        print(e)
        return
    
    if is_compensated:
        step = 0.2
    else:
        step = np.array([0.1325, 0.1645, 0.1842, 0.1896, 0.1907, 0.1963, 0.2])
    # step = 0.2

    kappa_s = -df['fit_singles'].to_numpy()/step
    kappa_s_err = df['fit_singles_err'].to_numpy()/step

    kappa_n = -df['fit_dens'].to_numpy() /step
    kappa_n_err = df['fit_dens_err'].to_numpy() / step

    # idxs = kappa_n_err ==0
    # kappa_n[idxs] = np.nan
    # kappa_n_err[idxs] = np.nan
    # kappa_n /=step
    # kappa_n_err/=step

    kappa_d = -df['fit_doublons'].to_numpy() / step
    kappa_d_err = df['fit_doublons_err'].to_numpy() / step
    # print(kappa_d_err)
    # idxs = kappa_d_err ==0
    # kappa_d[idxs] = np.nan
    # kappa_d_err[idxs] = np.nan
    # kappa_d /= step
    # kappa_d_err /= step
    
    periods = df['periods'].to_numpy()
    wavevectors = 1/periods
    doping = 1-density
    ax_[0].errorbar(wavevectors, kappa_s, kappa_s_err, label=dataset)
    ax_[1].errorbar(wavevectors, kappa_n, kappa_n_err, label=dataset)
    ax_[2].errorbar(wavevectors, kappa_d, kappa_d_err, label=dataset)
    
    return doping, wavevectors, kappa_s, kappa_s_err, kappa_n, kappa_n_err, kappa_d, kappa_d_err



with open('scan_group_info.yaml', "r") as file:
    scan_groups = yaml.safe_load(file)
print(list(scan_groups.keys()))

dopings_list = []
wavevectors_list = []
kappa_s_list = []
kappa_n_list = []
kappa_d_list = []
kappa_s_err_list = []
kappa_n_err_list = []
kappa_d_err_list = []
dset_list = []
# print(scan_groups['U8_Oct_6_p_83'])
for i,dataset in enumerate(scan_groups):
    # if dataset == "":
    if dataset in ['U8_Oct_18_p_83','U8_Oct_19_p_83_hot', 'lowU_Oct_11']:
        print('plotting {}'.format(dataset))
        if dataset == 'U8_Oct_21_p_83_cold':
            compensated = True
        else:
            compensated = False
        d, wvs, kappa_s, kappa_s_err, kappa_n, kappa_n_err, kappa_d, kappa_d_err = get_vals(dataset, ax_, compensated)
        dset_list.append(dataset)
        dopings_list.append(d)
        wavevectors_list.append(wvs)
        kappa_s_list.append(kappa_s)
        kappa_n_list.append(kappa_n)
        kappa_d_list.append(kappa_d)
        kappa_s_err_list.append(kappa_s_err)
        kappa_n_err_list.append(kappa_n_err)
        kappa_d_err_list.append(kappa_d_err)

for i in range(3):
    ax_[i].set_xlabel('Wavevector')
    ax_[i].set_position([0.15, 0.075+0.31*(2-i), 0.55, 0.27])
ax_[0].set_ylim(0, 0.3)
ax_[1].set_ylim(0, 0.4)
ax_[2].set_ylim(0, 0.1)
ax_[0].legend(loc=2, bbox_to_anchor=(1,1))
ax_[0].set_ylabel('kappa_singles')
ax_[1].set_ylabel('kappa_density')
ax_[2].set_ylabel('kappa_doublons')
# plt.show()


print(dset_list)
hot = 2
cold = 1
low = 0

def whiteblend(color,alpha):
    return [(1-alpha)+alpha*c for c in color]

fig, ax = plt.subplots(figsize=(3,2))
ax.errorbar(wavevectors_list[hot], kappa_n_list[hot], yerr=kappa_n_err_list[hot], label='Hot', ls='-', marker='o', markeredgecolor='r', markerfacecolor=whiteblend(to_rgb('r'), 0.5), c='r')
ax.errorbar(wavevectors_list[cold], kappa_n_list[cold], yerr=kappa_n_err_list[cold], label='Cold', ls='-', marker='o', markeredgecolor='b', markerfacecolor=whiteblend(to_rgb('b'), 0.5), c='b')
ax.errorbar(wavevectors_list[low], kappa_n_list[low], yerr=kappa_n_err_list[low], label='Low U', ls='--', marker='o', markeredgecolor='k', markerfacecolor=whiteblend(to_rgb('k'), 0.5), c='k', alpha=0.25)
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Susceptibility')

ax.legend()

fig.savefig('plots/cold_hot_lowU.pdf', transparent=True, dpi=600, bbox_inches='tight')


plt.show()