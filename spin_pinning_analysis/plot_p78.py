import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from uncertainties import unumpy as unp, ufloat
import yaml

try:
    from cat3.methods.plotting import errorplot_y
except:
    from cat3.cat3.methods.plotting import errorplot_y

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

# runlog = '/Volumes/share/FileServer/RunLog'
runlog = 'W:/RunLog'

with open('scan_group_info.yaml', "r") as file:
    scan_groups = yaml.safe_load(file)

from global_pars import blowvar, blowdetvar, intvar, patvar, bivar, xbar, densvar, tempvar, yxCentersite, AACrop, center, num_low, num_high, p_avg_weight, nx, ny

datasets = ['Jun_24_pd8_vp0p25', 'Jun_17_pd8_vp0p5', 'Jun_16_pd8_vp1p5']

fig, ax = plt.subplots(ncols=len(datasets))
for ids, ds in enumerate(datasets):
    dvars = np.load('processed/{}_densvar.npy'.format(ds))
    p = np.load('processed/{}_singles_avg.npy'.format(ds))
    p = unp.uarray(p[:,0],p[:,1])
    d = np.load('processed/{}_doublon_avg.npy'.format(ds))
    d = unp.uarray(d[:,0],d[:,1])
    fd = np.load('processed/{}_doublon_fidelity.npy'.format(ds))
    ufd = ufloat(fd[0], fd[1])
    n = p+2*d/ufd
    errorplot_y(dvars, n, ax[ids])
    ax[ids].set_title(ds)
    print(ds)
    print(dvars)

powers = [2.4e-3, 2.2e-3, 2.3e-3] # 2.4e-3 also ok for last one, 2.1e-3 okish for second last
inds = [3, 1, 0] # 1 ok for last, 0 ok for first

# plot density and corrs
panelsize = 1
fig, ax = plt.subplots(nrows=len(datasets), ncols=2, figsize=(2*panelsize, len(datasets)*panelsize))
X0 = 8
X1 = 31
plot_roi = np.s_[X0:X1,X0:X1]
for ids, ds in enumerate(datasets):
    # plot density
    pmap = np.load('processed/{}_singles_map.npy'.format(ds))
    pmap = pmap[inds[ids]]
    dmap = np.load('processed/{}_doublon_map.npy'.format(ds))
    dmap = dmap[inds[ids]]
    fd = np.load('processed/{}_doublon_fidelity.npy'.format(ds))[0]
    nmap = pmap + 2*dmap/fd
    ax[ids,0].imshow(nmap[plot_roi], cmap='viridis', vmin=0, vmax=1.05)
    # plot CLr
    CLr = np.load('processed/{}_CLr.npy'.format(ds))
    print(CLr.shape)
    CLr = CLr[inds[ids]][0]
    vm = np.max(np.abs(CLr))
    print(vm)
    ax[ids,1].imshow(CLr[plot_roi], vmin=-vm, vmax=vm, cmap='seismic')
    # plot the ROIs
    shift_center = scan_groups[ds]['shift_center']
    period = scan_groups[ds]['period']
    center1 = [center[0]+shift_center[0], center[1]+shift_center[1]]
    strpL = center1[1]-period*0.75
    strpR = center1[1]-period*0.25
    Xmax = (10**2-(period/2)**2)**0.5-1
    strpU = center1[1]-Xmax
    strpD = center1[1]+Xmax
    linekwg = {'c': 'k', 'alpha': 0.3}
    strpL -= X0
    strpR -= X0
    strpU -= X0
    strpD -= X0
    ax[ids,1].plot([strpL, strpR, strpR, strpL, strpL], [strpU, strpU, strpD, strpD, strpU], **linekwg)
    ax[ids,0].plot([strpL, strpR, strpR, strpL, strpL], [strpU, strpU, strpD, strpD, strpU], **linekwg)
    strpL = center1[1]+period*0.75
    strpR = center1[1]+period*0.25
    strpL -= X0
    strpR -= X0
    ax[ids,1].plot([strpL, strpR, strpR, strpL, strpL], [strpU, strpU, strpD, strpD, strpU], **linekwg)
    ax[ids,0].plot([strpL, strpR, strpR, strpL, strpL], [strpU, strpU, strpD, strpD, strpU], **linekwg)
    for i in range(2):
        ax[ids, i].set_xticks([])
        ax[ids, i].set_yticks([])
    #     stripe_peak = center1[1]-period/2
    # xx, yy = array_coordinates_2d((nx, ny))
    # Xmax = (10**2-(period/2)**2)**0.5-1
    # beta = 5
    # mask = (np.abs(yy-stripe_peak)<period/4)*1/(1+np.exp(beta*(np.abs(xx-center1[0])-Xmax)))
    # ampl = -np.cos(2*np.pi*(yy-center1[1])/period)
    # sign = (-1)**(xx+yy)
    # weights = mask * ampl * sign
    # mask1 = (np.abs(yy-stripe_peak-period)<period/4)*1/(1+np.exp(beta*(np.abs(xx-center1[0])-Xmax)))
# ax[0,0].set_title(r'$n(\vec{r})$')
# ax[0,1].set_title(r'$C_{zz}(L,\vec{r})$')

fig.savefig('plots/density_CLr.pdf', transparent=True, bbox_inches='tight', dpi=600)

# plot pca eigs
neig = 4
fig, ax = plt.subplots(ncols=neig, nrows=len(datasets), figsize=(neig*panelsize, len(datasets)*panelsize))
for ids, ds in enumerate(datasets):
    PCAv = np.load('processed/{}_PCAv.npy'.format(ds))
    PCAv = PCAv[inds[ids]]
    PCAv = PCAv[:neig]
    # vm = np.max(np.abs(PCAv))
    vm = 0.2
    PCAw = np.load('processed/{}_PCAw.npy'.format(ds))
    PCAw = PCAw[inds[ids]]
    for i in range(neig):
        ax[ids, i].imshow(PCAv[i], cmap='seismic', vmin=-vm, vmax=vm)
        ax[ids, i].set_xticks([])
        ax[ids, i].set_yticks([])
        ax[ids, i].set_title(r'$\lambda={:.2f}$'.format(PCAw[i]))

fig.savefig('plots/PCA.pdf', transparent=True, bbox_inches='tight', dpi=600)
# fig.suptitle('Spin corr. PCA')


# plot CLR and PCA eig
neig = 2
CLRs = np.zeros((2, len(datasets)))
Vp = np.zeros(len(datasets))
PCA = np.zeros((len(datasets), neig))
for ids, ds in enumerate(datasets):
    # get CLR
    CLR = np.load('processed/{}_CLR_avg.npy'.format(ds))
    CLR = CLR[inds[ids]]
    CLRs[:, ids] = CLR
    # get PCA
    PCAw = np.load('processed/{}_PCAw.npy'.format(ds))
    PCAw = PCAw[inds[ids]]
    PCA[ids,:] = PCAw[:neig]
    # get Vp
    vp = scan_groups[ds]['Vpin']
    Vp[ids] = vp
fig, ax = plt.subplots(nrows=2, figsize=(1.6, 2.6))
ax[0].errorbar(Vp, CLRs[0, :], yerr=CLRs[1,:], ls='-', marker='o', color='b')
ax[0].axhline(0, ls='--', c='k', alpha=0.5)
ax[0].set_title(r'$C_{zz}(L,R)$')
colors = ['b', 'r']
for i in range(neig):
    ax[1].plot(Vp, PCA[:, i], label='{}'.format(i), ls='-', marker='o', color=colors[i])
ax[1].set_title('PCA eigenvalue')
ax[1].legend()
ax[0].set_xticklabels([])
ax[1].set_xlabel(r'$V_{pin}$')

fig.savefig('plots/CLR_PCA_vals.pdf', transparent=True, bbox_inches='tight', dpi=600)

plt.show()



