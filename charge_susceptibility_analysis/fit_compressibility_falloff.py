import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy as unp, ufloat
from scipy.optimize import curve_fit
import time
import os
from scipy.interpolate import interp1d, griddata

import yaml



# df_exp_cal = pd.read_csv("W:\\RunLog\\2025\\exp_n_to_ns_590.0.csv")
df_exp_cal = pd.read_csv(os.path.join('/Volumes','share','FileServer','RunLog','2025','exp_n_to_ns_590.0.csv'))
ns_to_n = interp1d(df_exp_cal['ns'], df_exp_cal['n'])

def fit_cutoff(dataset, is_compensated=False):
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

    kappa_n = -df['fit_dens'].to_numpy() /step
    kappa_n_err = df['fit_dens_err'].to_numpy() / step
    
    periods = df['periods'].to_numpy()
    wavevectors = 1/periods
    doping = 1-density
    # fig, ax = plt.subplots()
    # ax.errorbar(wavevectors, kappa_n, kappa_n_err, label=p)
    # ax.set_title(dataset)
    # ax.set_ylim([0,0.3])
    # try to fit something
    def fitfunc(k, k0, amp, width, amp2):
        return amp/(1+np.exp((k-k0)/width))+amp2
    p0 = [1/8, 0.2, 0.01, 0]
    bounds = ([0, 0, 0, -0.5], [np.max(wavevectors), 0.5, 0.1, 0.5])
    # def fitfunc(k, k0, amp, width):
    #     return amp/(1+np.exp((k-k0)/width))
    # p0 = [1/8, 0.2, 0.01]
    # bounds = ([0, 0, 0], [np.max(wavevectors), 0.5, 0.1])
    try:
        popt, pcov = curve_fit(fitfunc, wavevectors, kappa_n, sigma=kappa_n_err, p0=p0, absolute_sigma=True, bounds=bounds)
        success = True
    except:
        popt = p0
        pcov = np.zeros((4,4))
        # pcov = np.zeros((3,3))
        success = False
    pstd = np.diag(pcov)**0.5
    fitted = fitfunc(wavevectors, *popt)
    # ax.plot(wavevectors, fitted, label='{}'.format(success))
    # ax.legend()
    # better error estimate somehow
    def LL(x):
        y = fitfunc(wavevectors, *x)
        loss = ((y-kappa_n)**2/kappa_n_err**2).sum()
        return loss
    fig, ax = plt.subplots()
    kvals = np.linspace(popt[0]-0.3, popt[0]+0.3, num=100)
    LLs = [LL(np.array([k, popt[1], popt[2], popt[3]])) for k in kvals]
    ax.plot(kvals, LLs)
    ax.set_title(dataset)
    return doping, wavevectors, kappa_n, popt[0], pstd[0], success

# to_plot = ['U8_Oct_21_p_83_cold', 'U8_Oct_10_p_65', 'U8_Oct_6_p_69', 'U8_Oct_5_p_81','U8_Oct_5_p_72','U8_Oct_3_p_78']
to_plot = ['U8_Oct_5_p_81','U8_Oct_3_p_78']
# to_plot = ['U8_Oct_21_p_83_cold', 'U8_Oct_6_p_69']
# plotinds = [3,2,4,1,0,5]
plotinds = [0,1]
# plotinds = [0,1,2,3,4,5]
figp, axp = plt.subplots(ncols=len(to_plot), figsize=(8,3))
def mle_cutoff(dataset, is_compensated=False, plotted=0):
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

    kappa_n = -df['fit_dens'].to_numpy() /step
    kappa_n_err = df['fit_dens_err'].to_numpy() / step
    
    periods = df['periods'].to_numpy()
    wavevectors = 1/periods
    doping = 1-density
    if dataset in to_plot:
        plotted2 = plotinds[plotted]
        axp[plotted2].errorbar(wavevectors, kappa_n, kappa_n_err, label='{:.2f}'.format(doping))
        axp[plotted2].set_title(dataset)
        axp[plotted2].legend()
        axp[plotted2].set_ylim([0,0.4])

    def fitfunc(k, k0, amp, width, amp2):
        return amp/(1+np.exp((k-k0)/width))+amp2
    def LL(x):
        y = fitfunc(wavevectors, *x)
        loss = ((y-kappa_n)**2/kappa_n_err**2).sum()
        return -0.5*loss
    # 1d meshes
    kvals = np.linspace(0,0.5, num=20)
    avals = np.linspace(0,0.4, num=20)
    wvals = np.linspace(0.01,0.1, num=10)
    a2vals = np.linspace(0,0.4, num=20)
    # high d mesh
    kk = kvals.reshape((-1,1,1,1)) * np.ones_like(avals).reshape((1,-1,1,1)) * np.ones_like(wvals).reshape((1,1,-1,1)) * np.ones_like(a2vals).reshape((1,1,1,-1))
    aa = np.ones_like(kvals).reshape((-1,1,1,1)) * avals.reshape((1,-1,1,1)) * np.ones_like(wvals).reshape((1,1,-1,1)) * np.ones_like(a2vals).reshape((1,1,1,-1))
    ww = np.ones_like(kvals).reshape((-1,1,1,1)) * np.ones_like(avals).reshape((1,-1,1,1)) * wvals.reshape((1,1,-1,1)) * np.ones_like(a2vals).reshape((1,1,1,-1))
    aa2 = np.ones_like(kvals).reshape((-1,1,1,1)) * np.ones_like(avals).reshape((1,-1,1,1)) * np.ones_like(wvals).reshape((1,1,-1,1)) * a2vals.reshape((1,1,1,-1))
    # evaluate
    x4d = np.array([kk.ravel(), aa.ravel(), ww.ravel(), aa2.ravel()]).T
    LLs = np.array([LL(x) for x in x4d]).reshape((20,20,10,20))
    # normalize
    LLs -= np.max(LLs)
    # pass to likelihood and integrate over the guys we don't want
    LLs = np.exp(LLs)
    LLs = np.sum(LLs, axis=(1,2,3))
    # back to LL
    LLs = np.log(LLs)
    LLs -= np.max(LLs)
    fig, ax = plt.subplots()
    ax.plot(kvals, LLs)
    ax.set_title(dataset)
    # convert LL to prob to return
    LLs = np.exp(LLs)
    LLs = LLs/np.sum(LLs)
    return doping, wavevectors, kappa_n, LLs



with open('scan_group_info.yaml', "r") as file:
    scan_groups = yaml.safe_load(file)
print(list(scan_groups.keys()))

# print(scan_groups['U8_Oct_6_p_83'])
all_dopings = []
all_wavevectors = []
all_kappa = []
dopings = []
LLs = []
plotted = 0
for i,dataset in enumerate(scan_groups):
    if dataset not in ['U8_Oct_12_p_83','U8_Oct_18_p_83','U8_Oct_19_p_83_hot', 'lowU_Oct_11','U8_Oct_9_p_58_v2','U8_Oct_10_p_85','U13_']:
        print('plotting {}'.format(dataset))
        if dataset == 'U8_Oct_21_p_83_cold':
            compensated = True
        else:
            compensated = False
        doping, wavevectors, kappa_n, LL = mle_cutoff(dataset, compensated, plotted)
        if dataset in to_plot:
            plotted += 1
        print(doping)
        print(wavevectors)
        print(kappa_n)
        all_dopings += len(wavevectors)*[doping]
        all_wavevectors += list(wavevectors.copy()[::-1])
        all_kappa += list(kappa_n.copy()[::-1])
        dopings.append(doping)
        LLs.append(LL)

all_dopings = np.array(all_dopings).reshape((-1,7))
all_wavevectors = np.array(all_wavevectors).reshape((-1,7))
all_kappa = np.array(all_kappa).reshape((-1,7))

order = np.argsort(all_dopings[:,0])
all_dopings = all_dopings[order,:]
all_wavevectors = all_wavevectors[order,:]
all_kappa = all_kappa[order,:]

fig, ax = plt.subplots()
ax.pcolormesh(all_dopings, all_wavevectors, all_kappa)
# ax.imshow(all_kappa)
# ax.imshow(all_wavevectors)
# ax.imshow(all_dopings)

x = np.linspace(0,np.max(all_wavevectors))
ax.plot(x, x, c='r', ls='--')
ax.plot(x, 2*x, c='r', ls='--')
ax.set_ylim([np.min(all_wavevectors), np.max(all_wavevectors)])


fig, ax = plt.subplots()
dopings = np.array(dopings)
order = np.argsort(dopings)
LLs = np.array(LLs)
dopings = dopings[order]
LLs = LLs[order,:]
kvals = np.linspace(0,0.5, num=20)
kk, dd = np.meshgrid(kvals, dopings)
ax.pcolormesh(dd, kk, LLs)



plt.show()