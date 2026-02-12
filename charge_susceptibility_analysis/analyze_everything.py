import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import watershed_ift
from uncertainties import unumpy as unp, ufloat
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit
import os
import yaml

try:
    from cat3.cat3.loading.loading import load_processed_scans
    from cat3.cat3.methods.plotting import errorplot_y, errorplot_xy
    from cat3.cat3.methods.postselection import postselect_with_df_by_low_high#(df, amat, low, high):
    from cat3.cat3.methods.postselection import postselect_with_df_by_mean_std_atnum#(df, amat,sigmas=2):
    from cat3.cat3.methods.util import array_coordinates_2d, coordinates_2d
    from cat3.cat3.computations.corrs_vs_density import pp_vs_dens_square
    from cat3.cat3.methods.binning import bins_fixed_size, radial_distances
except:
    from cat3.loading.loading import load_processed_scans
    from cat3.methods.plotting import errorplot_y, errorplot_xy
    from cat3.methods.postselection import postselect_with_df_by_low_high#(df, amat, low, high):
    from cat3.methods.postselection import postselect_with_df_by_mean_std_atnum#(df, amat,sigmas=2):
    from cat3.methods.util import array_coordinates_2d, coordinates_2d
    from cat3.computations.corrs_vs_density import pp_vs_dens_square
    from cat3.methods.binning import bins_fixed_size, radial_distances

# runlog = '/Volumes/share/FileServer/RunLog'
runlog = 'W:/RunLog'
repo_path = ''
load_dir = 'data'

with open('scan_group_info.yaml', "r") as file:
    scan_groups = yaml.safe_load(file)


blowvar = 'LocalVariables.ImagingBlowOutID'
intvar = 'GlobalLatticeVariables2.PhysicsSplitRadialLoadingField'
patvar = 'GlobalLatticeVariables2.Dmd0SecondPatternFilename'
bivar = 'GlobalLatticeVariables2.IsImageBIFormation'
xbar = 'LocalVariables.XbarFrequency'

# variables
densvar = 'GlobalLatticeVariables2.ThirdDMD1RampPower'
tempvar = 'GlobalLatticeVariables2.LoadingGradientLatticeHoldingTime'

# parameters
d_bucket = 10



# parameters for counts analysis

yxCentersite = (82,79)
AACrop = (5,-4,5,-4)
shift_center = (1,1)
center = [40+shift_center[0],40+shift_center[1]]

# parameters for density/thermometry analysis
bucket_rad = 10
num_low = {0: 300, 1: 150, 2: 150}
num_high = {0: 500, 1: 250, 2: 250}
p_avg_weight = 2 / 3 # for 1:1:1

# convenient for later
exp_quantities = ['singles', 'doublons', 'density', 'nshots',
                  'singles_therm', 'corr_therm', 
                  'imaging_fidelity', 'doublon_fidelity', ]
exp_pars = [densvar, intvar, tempvar, 'period']

# def add_img_info_to_df(df_all):
#     print('start adding')
#     scans = np.unique(df_all['scanstring'])
#     quantities = ['dmd_label1_proc','dmd_label2_proc','img_diff','img_rim']
#     defaultvals = ['missing','misssing',-1,-1]
#     # dtypes = [str,str,float,float]
#     values = {}
#     for iq, q in enumerate(quantities):
#         values[q] = np.array(len(df_all)*[defaultvals[iq]])
#     for isc, scstr in enumerate(scans):
#         dfimg = pd.read_csv('processed/{}_vals.csv'.format(scstr))
#         dfsub = df_all.loc[df_all['scanstring']==scstr]
#         shots1 = dfsub['shot_idx'].to_numpy()
#         shots2 = dfimg['shot_idx'].to_numpy()
#         matching = shots1.reshape((-1,1)) == shots2.reshape((1,-1))
#         for q in quantities:
#             qout = np.matmul(matching, dfimg[q].to_numpy())
#             values[q][dfsub.index] = qout
#     for q in quantities:
#         df_all[q] = values[q]
#     print('done adding')
#     return df_all

def ad_hoc_postselection(df):
    # use for things like lasers unlocking
    # bad = (df['scanstring'] == '20250408-0026')*(df['shot_idx']<1800)*(df['shot_idx']>1700) # pump unlocked
    # df = df.loc[~bad]
    return df

def get_gradient_calibration(scanstrings):
    print('analyzing gradient with {}'.format(scanstrings))
    # load things
    df_all, amat_all = load_processed_scans(scanstrings, roi_hwidth=45, yx0=yxCentersite, dmd_fixer=True, runlog_path=runlog, load_direc=load_dir)
    amat_all = amat_all[:, AACrop[0]:AACrop[1], AACrop[2]:AACrop[3]]
    # df_all = add_img_info_to_df(df_all)
    # df_all = ad_hoc_postselection(df_all)
    # df_all = df_all.loc[df_all['LocalVariables.DMDID'] == 0] # drop shots without cleanup
    # inds = np.array([('bal' in p) for p in df_all[patvar]])
    inds = df_all[bivar]==1
    df_all = df_all.loc[~inds]
    amat_all = amat_all[df_all.index]
    df_all = df_all.reset_index(drop=True)
    
    nz, nx, ny = amat_all.shape
    xx = np.arange(nx).reshape((-1, 1)) * np.ones(ny).reshape((1, -1))
    yy = np.arange(ny).reshape((1, -1)) * np.ones(nx).reshape((-1, 1))
    distances = ((xx-center[0])**2+(yy-center[1])**2)**0.5

    roi_bucket = (distances <= d_bucket)

    dens = amat_all[(df_all[blowvar]==0)].mean(axis=0)
    dens_err = amat_all[(df_all[blowvar]==0)].std(axis=0)/amat_all[(df_all[blowvar]==0)].shape[0]**0.5
    # now fit gradient
    def fitfunc(xfake, d0, a, b):
        return d0+a*xx+b*yy
    p0 = [dens.mean(), 0, 0]
    popt, pcov = curve_fit(
                    lambda x, d0, a, b: fitfunc(x, d0, a, b)[roi_bucket],
                    xx[roi_bucket],
                    dens[roi_bucket],
                    sigma=dens_err[roi_bucket],
                    p0=p0,
                    absolute_sigma=True
                    )
    pstd = np.diag(pcov)**0.5
    return popt, pstd

#=====================================================================

def get_doublon_fidelity(scanstrings):
    print('getting doublon fidelity for {}'.format(scanstrings))
    df_all, amat_all = load_processed_scans(scanstrings, roi_hwidth=45, yx0=yxCentersite, dmd_fixer=True, runlog_path=runlog, load_direc=load_dir)
    amat_all = amat_all[:, AACrop[0]:AACrop[1], AACrop[2]:AACrop[3]]

    df_all = df_all.loc[(df_all[bivar] ==True)]
    amat_all = amat_all[df_all.index]
    df_all = df_all.reset_index(drop=True)

    df_dens = df_all
    amat_dens = amat_all

    nz, nx, ny = amat_all.shape
    xx = np.arange(nx).reshape((-1,1))*np.ones(ny).reshape((1,-1))-nx//2 - shift_center[0]
    yy = np.arange(ny).reshape((1,-1))*np.ones(nx).reshape((-1,1))-ny//2 - shift_center[1]
    dists = (xx ** 2 + yy ** 2) ** 0.5
    roi_dens = dists < 10

    # pick odd vs even based on atnum
    dfx = df_dens[(df_dens[blowvar]==1)]
    odd = (np.rint(xx+shift_center[0]+yy+shift_center[1]))%2 ==1
    even = (np.rint(xx+shift_center[0]+yy+shift_center[1]))%2 ==0
    roi_odd = roi_dens*odd
    roi_even = roi_dens*even
    deven = np.mean(np.mean(amat_dens[dfx.index], axis=0)[roi_even])
    dodd = np.mean(np.mean(amat_dens[dfx.index], axis=0)[roi_odd])
    roi_dens = roi_odd if dodd > deven else roi_even
    
    df_dens['dummy'] = 1
    dens_nums = np.sum(amat_dens*roi_dens.reshape((1,nx,ny)), axis=(1,2))
    nroi_dens = np.sum(roi_dens)
    df_dens['num'] = dens_nums

    tmp = df_dens[['shot_idx',blowvar,'num','dummy']]
    tmp_mean = tmp.groupby([blowvar],as_index=False).mean()
    tmp_sum = tmp.groupby([blowvar],as_index=False).sum()
    tmp_std = tmp.groupby([blowvar],as_index=False).std()
    dens_vals = tmp_mean['num'].to_numpy()/nroi_dens
    dens_errs = tmp_std['num'].to_numpy()/nroi_dens/tmp_sum['dummy'].to_numpy()**0.5

    udens = [ufloat(d,e) for d,e in zip(dens_vals, dens_errs)]
    print([f"{ud:.S}" for ud in udens])
    ufidelity = (udens[1]-udens[0]/2)/(1-udens[0])
    # ufidelity = udens[1]/(1-udens[0])
    print(f'fidelity= {ufidelity:.S}')
    return ufidelity

#=====================================================================
def get_imaging_fidelity(scanstrings):
    imaging_pulse_var = "GlobalImagingVariables.RamanImagingBetweenFramesNumberOfPulseTrains"

    df, amat = load_processed_scans(scanstrings, roi_hwidth=None, dmd_fixer=False, runlog_path=runlog, load_direc=load_dir)
    AACrop = (40,-40,40,-40)
    amat = amat[:, :, AACrop[0]:AACrop[1], AACrop[2]:AACrop[3]]
    diff_matrix = 2*amat[:,0] - amat[:,1]

    pulses = sorted(np.unique(df[imaging_pulse_var].to_numpy()))
    npulses = len(pulses)
    stay, stay_err = np.zeros(npulses), np.zeros(npulses)
    hop, hop_err = np.zeros(npulses), np.zeros(npulses)
    lost, lost_err = np.zeros(npulses), np.zeros(npulses)
    counts, counts_err = np.zeros(npulses), np.zeros(npulses)
    for i, var in enumerate(pulses):
        dfd = df.loc[(df[imaging_pulse_var]==var)]
        ntot = amat[dfd.index,0, :,:].sum(axis=(1,2))
        lost_list = np.sum((diff_matrix == 2)[dfd.index, :, :], axis=(1, 2)) / ntot
        lost[i] = np.mean(lost_list)
        lost_err[i] = np.std(lost_list) / np.sqrt(len(lost_list))

        stay_list = np.sum((diff_matrix == 1)[dfd.index, :, :], axis=(1, 2)) / ntot
        stay[i] = np.mean(stay_list)
        stay_err[i] = np.std(stay_list) / np.sqrt(len(stay_list))

        hop_list = np.sum((diff_matrix == -1)[dfd.index, :, :], axis=(1, 2)) / ntot
        hop[i] = np.mean(hop_list)
        hop_err[i] = np.std(hop_list) / np.sqrt(len(hop_list))

    stay_err[stay_err==0] = 1
    lost_err[lost_err==0] = 1
    hop_err[hop_err==0] = 1

    # fig, ax = plt.subplots()
    # # plot only no fit
    # ax.errorbar(pulses, stay-1, stay_err, fmt='o',label='stay-1')
    # ax.errorbar(pulses, lost, lost_err, fmt='o', label='loss')
    # ax.errorbar(pulses, hop, hop_err, fmt='o', label='hop')
    # ax.grid()
    # # plt.ylim(-0.05,0.05)
    # ax.axhline(0, c='k')
    # ax.legend()
    # ax.set_xlabel(scan_var)
    # ax.set_title(scanstrings)
    # ax.set_ylim(-0.15, 0.15)
    # pulses2 = np.linspace(np.min(pulses), np.max(pulses), 21)


    def linfit(x, c, m):
        return m*x+c

    fit_func = linfit
    p0 = [1, -1e-6]
    try:
        p=p0
        # print(p0)
        popt, pcov=curve_fit(fit_func, pulses, stay, sigma=stay_err, absolute_sigma=True, p0=p, bounds=([0,-1e3],[1.01,0]))
        # print(popt)
        uc = ufloat(popt[0], np.sqrt(pcov[0,0]))
        uval = ufloat(popt[1], np.sqrt(pcov[1,1]))
        # ax.plot(pulses2, fit_func(pulses2, *popt)-1, '--', c='C0',label=f'stay-1: {uc*100:.2S}%, val = {uval*1.5e5:.2S} % per image')
        ustay = uc

        p = [0]+[-p0[1]]
        popt, pcov=curve_fit(fit_func, pulses, lost, sigma=lost_err, absolute_sigma=True, p0=p, bounds=([0,0],[1,1e-3]))
        # print(popt)
        uc = ufloat(popt[0], np.sqrt(pcov[0,0]))
        uval = ufloat(popt[1], np.sqrt(pcov[1,1]))
        ulost = uc
        # ax.plot(pulses2, fit_func(pulses2, *popt), '--', c='C1', label=f'loss: {uc*100:.2S}%, val = {uval*1.5e5:.2S} % per image')

        p = [0]+[-p0[1]]
        popt, pcov=curve_fit(fit_func, pulses, hop, sigma=hop_err,absolute_sigma=True, p0=p, bounds=([0,0],[1,1e-3]))
        # print(popt)
        uc = ufloat(popt[0], np.sqrt(pcov[0,0]))
        uval = ufloat(popt[1], np.sqrt(pcov[1,1]))
        uhop = uc
        # ax.plot(pulses2, fit_func(pulses2, *popt), '--', c='C2', label=f'hop: {uc*100:.2S}%, val = {uval*1.5e5:.2S} % per image')
        # ax.legend(ncol=2, loc='upper right')
    except Exception as e:
        print("Caught exception ", e)
        print("Skipping fit")
        return ufloat(1.0, 0.0)
    print(f"Imaging fidelities: \nStay {ustay*100:.2S}%, Lost: {ulost*100:.2S}%, Hop: {uhop*100:.2S}%")
    return ustay

#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================

def get_density_diff(scanstrings, Bfield, is_doublon_resolved, uimaging_fidelity, label='test'):
    print('analyzing density_diff with {}'.format(scanstrings))
    print(Bfield, is_doublon_resolved, uimaging_fidelity, label)
    # load things
    df, amat = load_processed_scans(scanstrings, roi_hwidth=45, yx0=yxCentersite, dmd_fixer=True, runlog_path=runlog, load_direc=load_dir)
    amat = amat[:, AACrop[0]:AACrop[1], AACrop[2]:AACrop[3]]
    print(len(df))
    # df = add_img_info_to_df(df)
    # df_all = ad_hoc_postselection(df_all)
    # df_all = df_all.loc[df_all['LocalVariables.DMDID'] == 0] # drop shots without cleanup

    inds = np.array([('pinning' not in p) for p in df[patvar]]) # remove thermometry shots
    df = df.loc[~inds]
    amat= amat[df.index]
    df = df.reset_index(drop=True)
    print(len(df))

    print(df[intvar].unique())
    df = df.loc[(df[intvar]==Bfield)*(df[bivar]==0)] # remove BI calibration shots and incorrect field shots
    amat = amat[df.index]
    df = df.reset_index(drop=True)
    print(len(df))

    dmd1_power = np.unique(df[densvar])
    assert len(dmd1_power)==1
    dmd1_power = dmd1_power[0]

    heating_time = np.unique(df[tempvar])
    assert len(heating_time)==1
    heating_time = heating_time[0]


    nz, nx, ny = amat.shape
    xx = np.arange(nx).reshape((-1, 1)) * np.ones(ny).reshape((1, -1))
    yy = np.arange(ny).reshape((1, -1)) * np.ones(nx).reshape((-1, 1))
    distances = ((xx-center[0])**2+(yy-center[1])**2)**0.5

    roi = (distances <= d_bucket)
    nroi = np.sum(roi)
    nums = np.sum(amat * roi.reshape((1, nx, ny)), axis=(1, 2))/nroi
    df['num'] = nums
    df['dummy'] = 1

    fig, ax = plt.subplots()
    ax.set_title(f'{label}')
    ax.plot(df[(df[blowvar]==0)]['num'].to_numpy(), label='no blow')
    if Bfield==557:
        bad_inds = (df[blowvar]==0)*(df['num']<0.4)
    else:
        bad_inds = (df[blowvar]==0)*(df['num']<0.5)
    amat = amat[~bad_inds]
    df = df.loc[~bad_inds].reset_index(drop=True)
    ax.plot(df[(df[blowvar]==0)]['num'].to_numpy(), label='no blow filtered')
    ax.plot(gaussian_filter(df[(df[blowvar]==0)]['num'].to_numpy(), 10), lw=2, label='no blow filtered (smooth)')
    if is_doublon_resolved:
        if is_doublon_resolved == 1:
            ax.plot(df[(df[blowvar]==1)]['num'].to_numpy(), label='blow1 (doublons)')
            bad_inds = (df[blowvar]==1)*(df['num']>0.5)
            amat = amat[~bad_inds]
            df = df.loc[~bad_inds].reset_index(drop=True)
            bad_inds = (df[blowvar]==1)*(df['num']<0.001)
            amat = amat[~bad_inds]
            df = df.loc[~bad_inds].reset_index(drop=True)
            ax.plot(df[(df[blowvar]==1)]['num'].to_numpy(), label='blow1 (doublons) filtered')
            ax.plot(gaussian_filter(df[(df[blowvar]==1)]['num'].to_numpy(), 10), lw=2,  label='blow1 (doublons) filtered (smooth)')
        elif is_doublon_resolved == 2:
            ax.plot(df[(df[blowvar]==1)]['num'].to_numpy(), label='blow1 (n/2)')
            bad_inds = (df[blowvar]==1)*(df['num']>0.6)
            amat = amat[~bad_inds]
            df = df.loc[~bad_inds].reset_index(drop=True)
            bad_inds = (df[blowvar]==1)*(df['num']<0.2)
            amat = amat[~bad_inds]
            df = df.loc[~bad_inds].reset_index(drop=True)
            ax.plot(df[(df[blowvar]==1)]['num'].to_numpy(), label='blow1 (n/2) filtered')
            ax.plot(gaussian_filter(df[(df[blowvar]==1)]['num'].to_numpy(), 10), lw=2, label='blow1 (n/2) filtered (smooth)')
    print(len(df))
    ax.set_ylabel('Density in ROI')
    ax.set_xlabel('shots')
    ax.legend()

    # # figure out dmd names - complain if there is ambiguity
    pattern_names = np.sort(np.unique(df[patvar]))
    print(pattern_names)
    pattern_names_n = pattern_names[[(pat[2:-1]).endswith("n.pkl") for pat in pattern_names]]
    pattern_names_p = pattern_names[[(pat[2:-1]).endswith("p.pkl") for pat in pattern_names]]
    periods = np.array([int((pat[2:-1]).split('_')[-1][:-5]) for pat in pattern_names_n])
    print(periods)

    norm_inds = (yy[0] - center[1] <= d_bucket - 1) & (yy[0] - center[1] >= -d_bucket + 1)
    nsites = np.sum(norm_inds)

    uint_dens_n = unp.uarray(np.zeros((2, len(periods), nsites)),
                             np.zeros((2, len(periods), nsites)))
    uint_dens_p = unp.uarray(np.zeros((2, len(periods), nsites)),
                             np.zeros((2, len(periods), nsites)))
    cov_n = np.zeros((2, len(periods), nsites,nsites))
    cov_p = np.zeros((2, len(periods), nsites,nsites))
    nshots_n = np.zeros((2, len(periods)))
    nshots_p = np.zeros((2, len(periods)))

    norm = np.sum(roi, axis=0).astype(float)
    norm[norm<=0] = np.nan
    shot_integrated = np.sum(amat*(roi.reshape((1, nx, ny)).astype(float)), axis=1)/norm.reshape(1, -1)
    shot_integrated = shot_integrated[:, norm_inds]
    shot_dens = np.sum(amat*(roi.reshape((1, nx, ny)).astype(float)), axis=(1,2))/nroi
    udens =  unp.uarray(np.zeros((2,len(periods))), np.zeros((2,len(periods))) )
    for ig, (pos, neg) in enumerate(zip(pattern_names_p, pattern_names_n)):
        for bid in range(2):
            dfp = df.loc[(df[patvar]==pos)&(df[blowvar]==bid)]
            dfn = df.loc[(df[patvar]==neg)&(df[blowvar]==bid)]
            dfboth = df.loc[(df[blowvar]==bid)&((df[patvar]==neg)|(df[patvar]==pos))]
            print(bid, ig, len(dfp), len(dfn), len(dfboth))

            if len(dfboth)>0:
                udens[bid, ig] = unp.uarray(np.mean(shot_dens[dfboth.index]), np.std(shot_dens[dfboth.index])/np.sqrt(len(dfboth)))

            if len(dfp)>0:
                nshots_p[bid, ig] = len(dfp)
                samples_p = shot_integrated[dfp.index]
                cov_p[bid,ig]= np.cov(samples_p.T)/len(dfp)
                uint_dens_p[bid,ig] = unp.uarray(np.mean(samples_p, axis=0),np.sqrt(np.diagonal(cov_p[bid,ig])))

            if len(dfn)>0:
                nshots_n[bid, ig] = len(dfn)
                samples_n = shot_integrated[dfn.index]
                cov_n[bid,ig]= np.cov(samples_n.T)/len(dfn)
                uint_dens_n[bid,ig] = unp.uarray(np.mean(samples_n, axis=0),np.sqrt(np.diagonal(cov_n[bid,ig])))
    
    uint_sing_n = uint_dens_n[0]/uimaging_fidelity.n
    uint_sing_p = uint_dens_p[0]/uimaging_fidelity.n
    uint_sing_diff = uint_sing_p-uint_sing_n
    cov_sing_diff = (cov_n[0]+cov_p[0])/uimaging_fidelity.n**2
    print(uint_sing_diff.shape)

    uns2  = udens[0]/uimaging_fidelity.n

    if is_doublon_resolved:
        udoublon_fidelity = get_doublon_fidelity(scanstrings)

        if is_doublon_resolved == 1:
            uint_doub_n = uint_dens_n[1]/udoublon_fidelity.n
            uint_doub_p = uint_dens_p[1]/udoublon_fidelity.n
            und = udens[1]/udoublon_fidelity.n
            uint_doub_diff = uint_doub_p-uint_doub_n            
            cov_doub_diff = (cov_n[1]+cov_p[1])/(udoublon_fidelity.n**2)
        elif is_doublon_resolved==2:
            uint_doub_n = (uint_dens_n[1]-uint_dens_n[0]/2)/udoublon_fidelity.n
            uint_doub_p = (uint_dens_p[1]-uint_dens_p[0]/2)/udoublon_fidelity.n
            und = (udens[1]-udens[0]/2)/udoublon_fidelity.n
            uint_doub_diff = uint_doub_p-uint_doub_n            
            cov_doub_diff = (cov_n[1]+cov_n[0]/4 +cov_p[1]+cov_p[0]/4)/(udoublon_fidelity.n**2)

        uint_dens_n = uint_sing_n + 2*uint_doub_n
        uint_dens_p = uint_sing_p + 2*uint_doub_p
        uint_dens_diff = uint_sing_diff + 2*uint_doub_diff
        cov_dens_diff = cov_sing_diff + 4*cov_doub_diff
        udensity = uns2 + 2*und

    else:
        uint_doub_n = -unp.uarray(np.ones((2, len(periods), nsites)),
                             np.zeros((2, len(periods), nsites)))
        uint_doub_p = -unp.uarray(np.ones((2, len(periods), nsites)),
                             np.zeros((2, len(periods), nsites)))
        uint_doub_diff = -unp.uarray(np.ones((len(periods), nsites)),
                             np.zeros((len(periods), nsites)))
        
        uint_dens_n = -unp.uarray(np.ones((2, len(periods), nsites)),
                             np.zeros((2, len(periods), nsites)))
        uint_dens_p = -unp.uarray(np.ones((2, len(periods), nsites)),
                             np.zeros((2, len(periods), nsites)))
        uint_dens_diff = -unp.uarray(np.ones((len(periods), nsites)),
                             np.zeros((len(periods), nsites)))
        und = -unp.uarray(np.ones(len(periods)),np.zeros(len(periods)))
        udensity = -unp.uarray(np.ones(len(periods)),np.zeros(len(periods)))
    
    fit_ampl = unp.uarray(np.zeros(len(periods)),np.zeros(len(periods)))
    fit_singles_ampl=unp.uarray(np.zeros(len(periods)),np.zeros(len(periods)))
    fit_doublons_ampl=unp.uarray(np.zeros(len(periods)),np.zeros(len(periods)))

    x = np.linspace(-d_bucket+1, d_bucket-1, nsites, endpoint=True)
    # x = yy[0][norm>0]-center[1]
    print(x.shape)
    # fig, ax = plt.subplots()
    # ax.set_title(label)
    for ip, period in enumerate(periods):
        def sine_fit(x, ampl):
            return ampl*np.sin(2*np.pi*x/period)
        # popt, pcov = curve_fit(sine_fit, x, unp.nominal_values(uint_sing_diff[ip]), sigma=unp.std_devs(uint_sing_diff[ip]), absolute_sigma=True)
        # ax.errorbar(x, unp.nominal_values(uint_sing_diff[ip]), unp.std_devs(uint_sing_diff[ip]), label=period, fmt='o', c=f"C{ip}")
        popt, pcov = curve_fit(sine_fit, x, unp.nominal_values(uint_sing_diff[ip]), sigma=cov_sing_diff[ip], absolute_sigma=True)
        # ax.plot(x, sine_fit(x,*popt), label=period, c=f"C{ip}")
        fit_singles_ampl[ip] = ufloat(popt[0], np.sqrt(pcov[0,0]))
        if is_doublon_resolved:
            # popt, pcov = curve_fit(sine_fit, x, unp.nominal_values(uint_dens_diff[ip]), sigma=unp.std_devs(uint_dens_diff[ip]), absolute_sigma=True)
            popt, pcov = curve_fit(sine_fit, x, unp.nominal_values(uint_dens_diff[ip]), sigma=cov_dens_diff[ip], absolute_sigma=True)
            fit_ampl[ip] = ufloat(popt[0], np.sqrt(pcov[0,0]))
            popt, pcov = curve_fit(sine_fit, x, unp.nominal_values(uint_doub_diff[ip]), sigma=unp.std_devs(uint_doub_diff[ip]), absolute_sigma=True)
            # popt, pcov = curve_fit(sine_fit, x, unp.nominal_values(uint_doub_diff[ip]), sigma=cov_doub_diff[ip], absolute_sigma=True)
            fit_doublons_ampl[ip] = ufloat(popt[0], np.sqrt(pcov[0,0]))
    ax.legend()
    print(periods.shape, nshots_n.shape, uns2.shape, und.shape, udensity.shape, fit_ampl.shape, fit_singles_ampl.shape, fit_doublons_ampl.shape)
    # construct output and return
    output = {densvar: np.round(dmd1_power,8)*np.ones(len(periods)),
            tempvar: np.round(heating_time, 8)*np.ones(len(periods)),
            'periods': periods,
            'nshots0_n': nshots_n[0],
            'nshots0_p': nshots_p[0],
            'nshots1_n': nshots_n[1],
            'nshots1_p': nshots_p[1],

            'avg_singles': unp.nominal_values(uns2), 
            'avg_singles_err': unp.std_devs(uns2), 
            'avg_doublons': unp.nominal_values(und), 
            'avg_doublons_err': unp.std_devs(und),
            'avg_dens': unp.nominal_values(udensity),
            'avg_dens_err': unp.std_devs(udensity),

            'fit_dens': unp.nominal_values(fit_ampl),
            'fit_dens_err': unp.std_devs(fit_ampl),
            'fit_singles': unp.nominal_values(fit_singles_ampl),
            'fit_singles_err': unp.std_devs(fit_singles_ampl),
            'fit_doublons': unp.nominal_values(fit_doublons_ampl),
            'fit_doublons_err': unp.std_devs(fit_doublons_ampl),
    }
    output2d = {densvar: np.round(dmd1_power,8)*np.ones(len(periods)),
            tempvar: np.round(heating_time, 8)*np.ones(len(periods)),
            'periods': periods,
            'nshots_n': nshots_n,
            'nshots_p': nshots_p,

            'diff_singles': unp.nominal_values(uint_sing_diff), 
            'diff_singles_err': unp.std_devs(uint_sing_diff), 
            'diff_doublons': unp.nominal_values(uint_doub_diff), 
            'diff_doublons_err': unp.std_devs(uint_doub_diff),
            'diff_dens': unp.nominal_values(uint_dens_diff),
            'diff_dens_err': unp.std_devs(uint_dens_diff),

            'singles_p': unp.nominal_values(uint_sing_p),
            'singles_p_err': unp.std_devs(uint_sing_p),
            'singles_n': unp.nominal_values(uint_sing_n),
            'singles_n_err': unp.std_devs(uint_sing_n),            
            'doublons_p': unp.nominal_values(uint_doub_p),
            'doublons_p_err': unp.std_devs(uint_doub_p),
            'doublons_n': unp.nominal_values(uint_doub_n),
            'doublons_n_err': unp.std_devs(uint_doub_n),
            'density_p': unp.nominal_values(uint_dens_p),
            'density_p_err': unp.std_devs(uint_dens_p),
            'density_n': unp.nominal_values(uint_dens_n),
            'density_n_err': unp.std_devs(uint_dens_n),
            }
    with open("processed/{}_output2d.npz".format(label), 'wb') as f:
        np.savez(f, **output2d)

    return output

#===================================================

def get_temperature(scanstrings, therm_pwr):
    print('analyzing temperature with {}'.format(scanstrings))
    # load
    df_all, amat_all = load_processed_scans(scanstrings, roi_hwidth=45, yx0=yxCentersite, dmd_fixer=True, runlog_path=runlog, load_direc=load_dir)
    amat_all = amat_all[:, AACrop[0]:AACrop[1], AACrop[2]:AACrop[3]]
    print(len(df_all))
    # postselect things out
    df_all = ad_hoc_postselection(df_all)
    df_all = df_all.loc[df_all[bivar] == 0] # no band insulator shots
    # df_all = df_all.loc[df_all[densvar] == therm_pwr] # thermometry at 2.4mW or 2.2mW
    df_all = df_all.loc[df_all[intvar] == 590] # 590G
    amat_all = amat_all[df_all.index]
    df_all = df_all.reset_index(drop=True)
    print(len(df_all))
    if len(df_all)==0:
        tmp = ufloat(0.0,1.0)
        return tmp, tmp, tmp, tmp, tmp, 1
    
    # figure out dmd names - complain if there is ambiguity
    pattern_names = np.unique(df_all[patvar])
    pattern_idx = ['pinning' not in pat for pat in pattern_names]
    pattern = pattern_names[pattern_idx][0]
    # remove any step pattern
    df_all = df_all.loc[(df_all[patvar]==pattern)]
    print(len(df_all))
    amat_all = amat_all[df_all.index]
    df_all = df_all.reset_index(drop=True)
    print(len(df_all))
    if len(df_all)==0:
        tmp = ufloat(0.0,1.0)
        return tmp, tmp, tmp, tmp, tmp, 1

    # crop for speed
    manual_roi = np.s_[:, 20:61, 20:61]
    amat_all = amat_all[manual_roi]
    nz, nx, ny = amat_all.shape
    center = [20+shift_center[0],20+shift_center[1]]
    xx = np.arange(nx).reshape((-1, 1)) * np.ones(ny).reshape((1, -1))
    yy = np.arange(ny).reshape((1, -1)) * np.ones(nx).reshape((-1, 1))
    distances = ((xx-center[0])**2+(yy-center[1])**2)**0.5
    roi_bucket = distances < d_bucket
    # get ROI, bins etc
    distances_2d = radial_distances(roi_bucket, metric=np.identity(2))  # 2d info
    bins, dists = bins_fixed_size(distances_2d, nbin=np.sum(distances_2d < bucket_rad), return_distances=True)

    temps = np.sort(np.unique(df_all[tempvar]))
    print(temps)
    p_avg_list = []
    ss_10_list = []
    ss_11_list = []
    ss_20_list = []
    ss_21_list = []
    # print('TEMPERATURE ANALYSIS')
    # print(len(df_all))
    for it, hold in enumerate(temps):
        dfx = df_all.loc[df_all[tempvar]==hold]
        # split by blowid and temps
        dfs = {bid: dfx.loc[dfx[blowvar] == bid] for bid in range(3)}
        amats = {bid: amat_all[dfs[bid].index] for bid in range(3)}
        # number postselection
        for bid in range(3):
            df, amat = dfs[bid], amats[bid]
            n1 = len(df)
            df, amat = postselect_with_df_by_low_high(df, amat, num_low[bid], num_high[bid])
            n2 = len(df)
            print('retain {}/{}'.format(n2, n1))
            dfs[bid], amats[bid] = df, amat
        
        # correlations
        p_all = {}
        ppc_10_all = {}
        ppc_11_all = {}
        ppc_20_all = {}
        ppc_21_all = {}
        for bid in range(3):
            amat = amats[bid]
            comp = pp_vs_dens_square(amat, nboot=500, verbose=False)
            comp.run(bins=bins)
            ppc_10_all[bid] = comp.A[comp.obshash['pp_c_s1'], :][0]
            ppc_11_all[bid] = comp.A[comp.obshash['pp_c_s2'], :][0]
            ppc_20_all[bid] = comp.A[comp.obshash['pp_c_s22'], :][0]
            ppc_21_all[bid] = comp.A[comp.obshash['pp_c_s21'], :][0]
            p_all[bid] = comp.A[comp.obshash['p'], :][0]
        ss_10_list.append(2 * (ppc_10_all[1] + ppc_10_all[2]) - ppc_10_all[0])
        ss_11_list.append(2 * (ppc_11_all[1] + ppc_11_all[2]) - ppc_11_all[0])
        ss_20_list.append(2 * (ppc_20_all[1] + ppc_20_all[2]) - ppc_20_all[0])
        ss_21_list.append(2 * (ppc_21_all[1] + ppc_21_all[2]) - ppc_21_all[0])
        p_avg_list.append( p_avg_weight * p_all[0] + (1-p_avg_weight)*(p_all[1]+p_all[2]))
    # print('TEMP ANALYSIS OUTPUT')
    # print(p_avg_list)
    # print(ss_10_list)
    nshot = len(df_all)
    if len(temps)==1:
        pavg = p_avg_list[0]
        ss10, ss11, ss20, ss21 = ss_10_list[0], ss_11_list[0], ss_20_list[0], ss_21_list[0]
        return pavg, ss10, ss11, ss20, ss21, nshot
    else:
        pavg = np.array(p_avg_list)
        ss10, ss11, ss20, ss21 = np.array(ss_10_list), np.array(ss_11_list), np.array(ss_20_list), np.array(ss_21_list)
        return pavg, ss10, ss11, ss20, ss21, np.array(len(temps)*[nshot//len(temps)])
#=================================================================


def analyze_scan_group(scan_group, dataset):
    print('analyzing {}'.format(scan_group['comment']))
    if 'thermometry' not in scan_group.keys():
        print("Using data scans for thermometry")
        scan_group['thermometry'] = scan_group['data']
    if 'Bfield' not in scan_group.keys():
        print("Using 590 G for Bfield")
        scan_group['Bfield'] = 590
        

    # run all the analyses
    uimaging_fidelity = get_imaging_fidelity(scan_group['imaging_calibration'])
    diff_data = get_density_diff(scan_group['data'], scan_group['Bfield'], 
                                 scan_group['is_doublon_resolved'], uimaging_fidelity,
                                 dataset)
    therm_data = get_temperature(scan_group['thermometry'], scan_group['thermometry_power'])
    gpopt, gpstd = get_gradient_calibration(scan_group['data'])
    
    # normalize things by imaging fidelity
    therm_data = np.array(therm_data)
    nshot_therm = therm_data[-1]
    if not isinstance(nshot_therm,int):
        nshot_therm = therm_data[-1].copy()
    therm_data[0] = therm_data[0]/uimaging_fidelity
    therm_data[1:] = therm_data[1:]/(uimaging_fidelity**2)

    npts = len(diff_data['periods'])
    print("npts: ", npts)
    # attach gradient calibration
    diff_data['grad0'] = npts * [gpopt[1]]
    diff_data['grad0_err'] = npts * [gpstd[1]]
    diff_data['grad1'] = npts * [gpopt[2]]
    diff_data['grad1_err'] = npts * [gpstd[2]]

    # attach imaging calibration
    diff_data['imaging_calibration'] = npts * [unp.nominal_values(uimaging_fidelity)]
    diff_data['imaging_calibration_err'] = npts * [unp.std_devs(uimaging_fidelity)]

    diff_data['Bfield'] = npts * [scan_group['Bfield']]

    # attach thermometry
    diff_data['singles_therm'] = npts * [unp.nominal_values(therm_data[0])]
    diff_data['singles_therm_err'] = npts * [unp.std_devs(therm_data[0])]
    diff_data['corr_therm'] = npts * [unp.nominal_values(therm_data[1])]
    diff_data['corr_therm_err'] = npts * [unp.std_devs(therm_data[1])]
    diff_data['nshot_therm'] = npts * [nshot_therm]

    # ntemps = len(np.unique(diff_data[tempvar]))
    # ndens = len(np.unique(diff_data[densvar]))    
    # if ntemps == 1:
    #     diff_data['singles_therm'] = ndens * [unp.nominal_values(therm_data[0])]
    #     diff_data['singles_therm_err'] = ndens * [unp.std_devs(therm_data[0])]
    #     diff_data['corr_therm'] = ndens * [unp.nominal_values(therm_data[1])]
    #     diff_data['corr_therm_err'] = ndens * [unp.std_devs(therm_data[1])]
    #     diff_data['nshot_therm'] = ndens * [nshot_therm]
    #     # print('ADDING TO CSV')
    #     # print(unp.nominal_values(therm_data[0]), unp.std_devs(therm_data[0]))
    #     # print(unp.nominal_values(therm_data[1]), unp.std_devs(therm_data[1]))
    # else:
    #     p_lst = []
    #     p_err_lst = []
    #     corr_lst = []
    #     corr_err_lst = []
    #     nshot_lst = []
    #     for it in range(ntemps):
    #         p_lst += ndens * [unp.nominal_values(therm_data[0][it])]
    #         p_err_lst += ndens * [unp.std_devs(therm_data[0][it])]
    #         corr_lst += ndens * [unp.nominal_values(therm_data[1][it])]
    #         corr_err_lst += ndens * [unp.std_devs(therm_data[1][it])]
    #         nshot_lst += ndens * [nshot_therm[it]]
    #     diff_data['singles_therm'] = p_lst
    #     diff_data['singles_therm_err'] = p_err_lst
    #     diff_data['corr_therm'] = corr_lst
    #     diff_data['corr_therm_err'] = corr_err_lst
    #     diff_data['nshot_therm'] = nshot_lst

    # for key in diff_data.keys():
    #     print(key, len(diff_data[key]))
        # len(diff_data[key][0]))
    # return
    return diff_data


# # # loop through and analyze
for dataset in scan_groups:
    # if dataset == "U8_Oct_3_p_78":
    # if dataset == "U8_Oct_9_p_58_v2":
    # if dataset == "U8_Oct_12_p_83":
    # if dataset == "lowU_Oct_11":
    # if dataset == "U8_Oct_10_p_65":
#     # if dataset in ['U8_hot_cold_step3_round1','U8_hot_cold_step3_round2','U8_hot_cold_step3_round3', 'U8_hot_cold_step3_round4']:
#     # if dataset in ['U3_cold_step3_round1', 'U3_cold_step3_round2']:
#     # if dataset in ['U5_cold_step3_round1', 'U5_cold_step3_round2']:
#     # if dataset in ['U4_mid_temp_step3', 'U4_cold_step3']:
#     # if dataset in ['U6_cold_step3_round1', 'U6_cold_step3_round2']:
        print('analyzing {}'.format(dataset))
        processed_data = analyze_scan_group(scan_groups[dataset], dataset)
        processed_data['dataset'] = len(processed_data[densvar])*[dataset]
        dfout = pd.DataFrame.from_dict(processed_data)
        dfout.to_csv(os.path.join(repo_path, 'processed', '{}.csv'.format(dataset)), index=False)
plt.show()


# # loop through and analyze all datasets scan resolved
# # RUN THIS ONE
# for dataset in scan_groups:
#     # if dataset == "U11_cold_step3":
#     # if dataset in ['U8_hot_cold_step3_round1','U8_hot_cold_step3_round2','U8_hot_cold_step3_round3', 'U8_hot_cold_step3_round4']:
#     # if dataset in ['U3_cold_step3_round1', 'U3_cold_step3_round2']:
#     # if dataset in ['U5_cold_step3_round1', 'U5_cold_step3_round2']:
#     # if dataset in ['U4_mid_temp_step3', 'U4_cold_step3']:
#     # if dataset in ['U6_cold_step3_round1', 'U6_cold_step3_round2']:
#     print('analyzing {}'.format(dataset))
#     df_list = []
#     for scstr in scan_groups[dataset]['data']:
#         nsg = {}
#         nsg['data'] = [scstr]
#         for key in scan_groups[dataset].keys():
#             if key != 'data':
#                 nsg[key] = scan_groups[dataset][key]
#         processed_data = analyze_scan_group(nsg)
#         processed_data['dataset'] = len(processed_data[densvar])*[dataset]
#         processed_data['scanstring'] = len(processed_data[densvar])*[scstr]
#         dfout = pd.DataFrame.from_dict(processed_data)
#         df_list.append(dfout)
#     df_out_all = pd.concat(df_list)
#     df_out_all.to_csv(os.path.join('processed', '{}_scan_resolved.csv'.format(dataset)), index=False)


# # scan resolved
# groups = [
#           # ['U3_cold_step3_round1', 'U3_cold_step3_round2'],
#           # ['U4_all'],
#           # ['U5_cold_step3_round1', 'U5_cold_step3_round2'],
#           # ['U6_cold_step3_round1', 'U6_cold_step3_round2'],
#           # ['U8_hot_cold_step3_round1', 'U8_hot_cold_step3_round2', 'U8_hot_cold_step3_round3', 'U8_hot_cold_step3_round4','U8_warm_step3'],
#           # ["U11_cold_step3"],
# ]
# #
# # # get all the data
# for U, group in zip([4], groups):
# # for U, group in zip([3,4,5,6,8,11], groups):
#     df_list = []
#     for dataset in group:
#         for scan in scan_groups[dataset]['data']:
#             nsg = {}
#             nsg['data'] = [scan]
#             for key in scan_groups[dataset].keys():
#                 if key != 'data':
#                     nsg[key] = scan_groups[dataset][key]
#             # print('HERE')
#             # print(nsg['data'])
#             processed_data = analyze_scan_group(nsg)
#             processed_data['scan'] = len(processed_data[densvar])*[scan]
#             # print('DONE WITH ANALYSIS')
#             # print(processed_data['singles_therm'],processed_data['singles_therm_err'])
#             # print(processed_data['corr_therm'],processed_data['corr_therm_err'])
#             dfout = pd.DataFrame.from_dict(processed_data)
#             df_list.append(dfout)
#     df_out_all = pd.concat(df_list)
#     df_out_all.to_csv(os.path.join(repo_path, 'processed', f'U{U}_all_scan_resolved.csv'), index=False)
# print('done')

