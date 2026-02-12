import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy as unp, ufloat
from scipy.optimize import curve_fit
import time
import os
from scipy.interpolate import interp1d, griddata

import yaml





fig_, ax_ = plt.subplots(nrows=3, sharex=True, figsize=(6, 8))


# df_exp_cal = pd.read_csv("W:\\RunLog\\2025\\exp_n_to_ns_590.0.csv")
df_exp_cal = pd.read_csv(os.path.join('/Volumes','share','FileServer','RunLog','2025','exp_n_to_ns_590.0.csv'))
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
# print(scan_groups['U8_Oct_6_p_83'])
for i,dataset in enumerate(scan_groups):
    # if dataset == "":
    if dataset not in ['U8_Oct_12_p_83','U8_Oct_18_p_83','U8_Oct_19_p_83_hot', 'lowU_Oct_11']:
    # for dataset in ['U8_Oct_6_p_83','U8_Oct_21_p_83_cold','U8_Oct_12_p_83','U8_Oct_18_p_83','U8_Oct_19_p_83_hot']:
        print('plotting {}'.format(dataset))
        if dataset == 'U8_Oct_21_p_83_cold':
            compensated = True
        else:
            compensated = False
        d, wvs, kappa_s, kappa_s_err, kappa_n, kappa_n_err, kappa_d, kappa_d_err = get_vals(dataset, ax_, compensated)
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
plt.show()

dopings_list = np.array(dopings_list)
wavevectors_list = np.array(wavevectors_list)
kappa_s_list = np.array(kappa_s_list)
kappa_n_list = np.array(kappa_n_list)
kappa_d_list = np.array(kappa_d_list)
kappa_s_err_list = np.array(kappa_s_err_list)
kappa_n_err_list = np.array(kappa_n_err_list)
kappa_d_err_list = np.array(kappa_d_err_list)

idxs = np.argsort(dopings_list)
dopings = dopings_list[idxs]
wavevectors = wavevectors_list[idxs]
kappa_s = kappa_s_list[idxs]
kappa_n = kappa_n_list[idxs]
kappa_d = kappa_d_list[idxs]
kappa_s_err = kappa_s_err_list[idxs]
kappa_n_err = kappa_n_err_list[idxs]
kappa_d_err = kappa_d_err_list[idxs]

print(dopings)
print(wavevectors.shape)

fig_, ax_ = plt.subplots(nrows=3, sharex=True, figsize=(6, 8))

for i in range(len(wavevectors[0])):
    ax_[0].errorbar(dopings, kappa_s[:,i], kappa_s_err[:,i], label=1/wavevectors[0,i])
    ax_[1].errorbar(dopings, kappa_n[:,i], kappa_n_err[:,i], label=1/wavevectors[0,i])
    ax_[2].errorbar(dopings, kappa_d[:,i], kappa_d_err[:,i], label=1/wavevectors[0,i])

for i in range(3):
    ax_[i].set_xlabel('Doping')
    ax_[i].set_position([0.15, 0.075+0.31*(2-i), 0.55, 0.27])
ax_[0].set_ylim(0, 0.3)
ax_[1].set_ylim(0, 0.4)
ax_[2].set_ylim(0, 0.1)
ax_[0].legend(loc=2, bbox_to_anchor=(1,1))
ax_[0].set_ylabel('kappa_singles')
ax_[1].set_ylabel('kappa_density')
ax_[2].set_ylabel('kappa_doublons')
plt.show()


def make_2d_plot(dopings, wavevectors, vals, vm=0.3):
    fig, ax = plt.subplots()
    for i in range(len(dopings)):
        if i==0:
            d_l = -0.01
            d_u = (dopings[0]+dopings[1])/2
        elif i==len(dopings)-1:
            d_l = (dopings[-1]+dopings[-2])/2
            d_u = dopings[-1]+0.05
        else:
            d_l = (dopings[i-1] + dopings[i])/2
            d_u = (dopings[i] + dopings[i+1])/2
        
        wvs = wavevectors[i]
        k = vals[i]     
        dopings2 = np.array([np.ones(len(wvs)+1)*d_l, 
                                np.ones(len(wvs)+1)*d_u])
        wavevectors2 = np.array([list(wvs+0.02)+[0], 
                                list(wvs+0.02)+[0]])
        # print(dopings2)
        # print(wavevectors2)
        im=ax.pcolormesh(dopings2.T, wavevectors2.T, k.reshape(-1,1), vmin=0, vmax=vm, cmap='viridis')
    ax.set_ylabel('Wavevector')
    ax.set_xlabel('Doping')
    fig.colorbar(im)
    return fig, ax


fig,ax = make_2d_plot(dopings, wavevectors, kappa_s, 0.3)
ax.set_title('Singles')
fig,ax = make_2d_plot(dopings, wavevectors, kappa_n, 0.4)
ax.set_title('Density')
x = np.linspace(0,1,num=20)
ax.plot(x, x, ls='--', c='r')
ax.plot(x, 2*x, ls='--', c='r')
ax.set_xlim([-0.01,0.41])
ax.set_ylim([0,0.27])

fig,ax = make_2d_plot(dopings, wavevectors, kappa_d, 0.1)
ax.set_title('Doublons')



plt.show()