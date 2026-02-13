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

dataset = 'Jun_22_pd8_vp1p5_vs_temp'
CLR = np.load('processed/{}_CLR_avg.npy'.format(dataset))
temps = np.load('processed/{}_tempvar.npy'.format(dataset))

fig, ax = plt.subplots()
ax.errorbar(temps, CLR[:,0], yerr=CLR[:,1])

PCA = np.load('processed/{}_PCAv.npy'.format(dataset))
vm = np.max(np.abs(PCA))
fig, ax = plt.subplots(nrows=5)
for i in range(5):
    ax[i].imshow(PCA[i,0], vmin=-vm, vmax=vm, cmap='seismic')

PCA = np.load('processed/{}_PCAw.npy'.format(dataset))
fig, ax = plt.subplots()
ax.plot(PCA[:, 0], label=0)
ax.plot(PCA[:, 1], label=1)
ax.legend()
plt.show()



