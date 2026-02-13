import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from uncertainties import unumpy as unp, ufloat
import yaml
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm

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

datasets = ['Jun_19_pd6_vp1', 'Jun_12_pd8_vp1', 'Jun_20_pd10_vp1']

# CLR map
dopings = []
periods = []
CLR = []
CLRe = []

for ids, ds in enumerate(datasets):
    # clr data and errors
    clr = np.load('processed/{}_CLR_avg.npy'.format(ds))
    CLR += list(clr[:, 0].ravel())
    CLRe += list(clr[:, 1].ravel())
    npt = clr.shape[0]
    # period
    period = scan_groups[ds]['period']
    periods += npt*[period]
    # density
    is_doublon = scan_groups[ds]['is_doublon_resolved']
    if is_doublon:
        singles = np.load('processed/{}_singles_avg.npy'.format(ds))
        singles = singles[:,0] # nom val
        doublons = np.load('processed/{}_doublon_avg.npy'.format(ds))
        doublons = doublons[:,0] # nom val
        fd = np.load('processed/{}_doublon_fidelity.npy'.format(ds))[0]
        density = singles+2*doublons/fd
    else:
        # something hacky: load singles map, crop to ROI, apply our conversion to it after a bit of smoothing
        singles_map = np.load('processed/{}_singles_map.npy'.format(ds))
        Xmax = 6
        shift_center = scan_groups[ds]['shift_center']
        center1 = [center[0]+shift_center[0], center[1]+shift_center[1]]
        roi_avg = np.s_[:, center1[0]-Xmax:center1[0]+Xmax+1, center1[1]-period:center1[1]+period]
        singles_map = singles_map[roi_avg]
        # singles_map = gaussian_filter(singles_map, sigma=0.5)
        singles_map = gaussian_filter(singles_map, sigma=0.75)
        # load conversion
        conv = pd.read_csv('aux/exp_n_to_ns_590.0.csv')
        convfunc = interp1d(conv['ns'], conv['n'])
        dens_map = convfunc(singles_map)
        density = dens_map.mean(axis=(1,2))
    dop = 1-density
    dopings += list(dop.ravel())

dopings = np.array(dopings)
periods = np.array(periods)
wavenumbers = 1/periods
CLR = np.array(CLR)
CLRe = np.array(CLRe)

def get_edges(arr):
    a1 = (arr[1:]+arr[:-1])/2
    lower = np.concatenate([np.array([arr[0]-(a1[0]-arr[0])]), a1])
    upper = np.concatenate([a1, np.array([arr[-1]+(arr[-1]-a1[-1])])])
    return lower, upper

def makeplot(ax, xx,yy,zz, zmin=0, zmax=30, cmap=cm.get_cmap('viridis')):
    yvals = np.unique(yy)
    ylower, yupper = get_edges(yvals)
    for iy, yv in enumerate(yvals):
        x = xx[yy==yv]
        z = zz[yy==yv]
        xlower, xupper = get_edges(x)
        print('HEREeeeee')
        print(iy,yv)
        print(x)
        print(xlower)
        print(xupper)
        for ix, xv in enumerate(x):
            xpts = [xlower[ix], xlower[ix], xupper[ix], xupper[ix]]
            ypts = [ylower[iy], yupper[iy], yupper[iy], ylower[iy]]
            ax.fill(xpts, ypts, color=cmap((z[ix]-zmin)/(zmax-zmin)), edgecolor=None)

panelsize = 3.4
fig, ax = plt.subplots(figsize=(panelsize, panelsize*0.8))
vm = np.max(np.abs(CLR))
makeplot(ax, dopings, wavenumbers, CLR, zmin=-vm, zmax=vm, cmap=cm.get_cmap('seismic'))
dnew = np.linspace(0, np.max(dopings), num=10, endpoint=True)
ax.plot(dnew, dnew, c='k', ls='--', alpha=0.5)
ax.plot(dnew, 2*dnew, c='k', ls='--', alpha=0.5)

# ax.set_xlim([0,0.22])
# ax.set_ylim([0,0.19])

ax.set_xlim([0.05,0.22])
ax.set_ylim([0.075,0.19])

ax.set_xlabel('Doping')
ax.set_ylabel('Wavenumber')

fig.savefig('plots/CLR_vs_dop_k.pdf', transparent=True, dpi=600, bbox_inches='tight')

plt.show()



