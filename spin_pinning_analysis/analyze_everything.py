import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy as unp, ufloat
import os
import yaml

try:
    from cat3.cat3.loading.loading import load_processed_scans
    from cat3.cat3.methods.plotting import errorplot_y, errorplot_xy
    from cat3.cat3.methods.postselection import postselect_with_df_by_low_high#(df, amat, low, high):
    from cat3.cat3.methods.postselection import postselect_with_df_by_mean_std_atnum#(df, amat,sigmas=2):
    from cat3.cat3.methods.util import array_coordinates_2d, coordinates_2d
    from cat3.cat3.computations.corrs_map import pp_correlation_map_site_chunk
    from cat3.cat3.computations.corrs_bucket import pp_correlation_map_chunk_chunk
except:
    from cat3.loading.loading import load_processed_scans
    from cat3.methods.plotting import errorplot_y, errorplot_xy
    from cat3.methods.postselection import postselect_with_df_by_low_high#(df, amat, low, high):
    from cat3.methods.postselection import postselect_with_df_by_mean_std_atnum#(df, amat,sigmas=2):
    from cat3.methods.util import array_coordinates_2d, coordinates_2d
    from cat3.computations.corrs_map import pp_correlation_map_site_chunk
    from cat3.computations.corrs_bucket import pp_correlation_map_chunk_chunk

# runlog = '/Volumes/share/FileServer/RunLog'
runlog = 'W:/RunLog'
repo_path = ''
load_dir = 'data'

with open('scan_group_info.yaml', "r") as file:
    scan_groups = yaml.safe_load(file)


from global_pars import blowvar, blowdetvar, intvar, patvar, bivar, xbar, densvar, tempvar, yxCentersite, AACrop, center, num_low, num_high, p_avg_weight, nx, ny



def ad_hoc_postselection(df):
    # use for things like lasers unlocking
    # bad = (df['scanstring'] == '20250408-0026')*(df['shot_idx']<1800)*(df['shot_idx']>1700) # pump unlocked
    # df = df.loc[~bad]
    return df

def get_doublon_fidelity(df, amat, center1, dataset):
    print('get doublon fidelity')
    # fig, ax = plt.subplots()
    # ax.plot(df[bivar])
    # print(np.max(df['shot_idx']))
    # # print(df[['shot_idx', 'scanstring', bivar]])
    # fig.suptitle(dataset)
    # plt.show()
    df = df.loc[df[bivar] == True]
    df0 = df.loc[df[blowvar] == 0]
    df1 = df.loc[df[blowvar] == 1]
    amat0 = amat[df0.index]
    amat1 = amat[df1.index]
    xx, yy = array_coordinates_2d((nx, ny))
    distance = ((xx-center1[0])**2+(yy-center1[1])**2)**0.5
    roi_doublon = distance<10
    if dataset == 'Jun_24_pd8_vp0p25':
        roi_doublon *= ((xx+yy)%2==0)
    else:
        roi_doublon *= ((xx+yy)%2==1)
    nroi = roi_doublon.sum()
    d0 = (amat0*roi_doublon.reshape((1, nx, ny))).sum(axis=(1,2))/nroi
    d1 = (amat1*roi_doublon.reshape((1, nx, ny))).sum(axis=(1,2))/nroi
    d0 = d0[d0<0.2]
    d1 = d1[d1>0.6]
    # fig, ax = plt.subplots()
    # ax.plot(d0)
    # ax.plot(d1)
    # fig.suptitle(dataset)
    # plt.show()
    m0 = d0.mean()
    s0 = d0.std()/len(d0)**0.5
    m1 = d1.mean()
    s1 = d1.std()/len(d1)**0.5
    um0 = ufloat(m0, s0)
    um1 = ufloat(m1, s1)
    fid = um1/(1-um0)
    # print(fid)
    return fid

def drop_doublon_calibration(df, amat):
    df = df.loc[df[bivar] == False]
    amat = amat[df.index]
    df = df.reset_index(drop=True)
    return df, amat

def atom_number_postselection(df, amat, is_doublon):
    # make a unified list to identify the blow id
    b_shots = np.zeros((4, len(df)))
    if is_doublon:
        b_shots[3] = (df[blowdetvar].to_numpy() == 610).astype(int)
    for bid in range(3):
        b_shots[bid] = (df[blowvar].to_numpy() == bid).astype(int)
        b_shots[bid] *= (1-b_shots[3]) # if doublon shot then not one of the others
    # on this basis assign what lower/upper lims shoudl be
    lowerlim = np.zeros(len(df))
    upperlim = np.zeros(len(df))
    for bid in range(4):
        lowerlim += b_shots[bid] * num_low[bid]
        upperlim += b_shots[bid] * num_high[bid]
    df['atom_lim_upper'] = upperlim
    df['atom_lim_lower'] = lowerlim
    # get atom number, compare to the limits, drop bad shots
    df['atom_num'] = amat.sum(axis=(1,2))
    good = (df['atom_num'] < df['atom_lim_upper'])*(df['atom_num'] > df['atom_lim_lower'])
    df = df.loc[good]
    amat = amat[df.index]
    df = df.reset_index(drop=True)
    return df, amat

def singles_map_analysis(df, amat, is_doublon):
    print('singles map analysis')
    # drop doublon shots, if they exist
    if is_doublon:
        df = df.loc[df[blowdetvar] != 610]
    # loop over densities/temperatures
    dvars = np.unique(df[densvar])
    # tvars = np.unique(df[tempvar])
    output = np.zeros((len(dvars), nx, ny))
    for idv, dv in enumerate(dvars):
        dfd = df.loc[(df[densvar]==dv)]
        dfs = {bid: dfd.loc[dfd[blowvar] == bid] for bid in range(3)}
        amats = {bid: amat[dfs[bid].index] for bid in range(3)}
        means = {bid: amats[bid].mean(axis=0) for bid in range(3)}
        output[idv] = p_avg_weight*means[0]+(1-p_avg_weight)*(means[1]+means[2])
    return dvars, output

def doublon_map_analysis(df, amat, is_doublon):
    print('doublon map analysis')
    # keep doublon shots
    df = df.loc[df[blowdetvar] == 610]
    # loop over densities/temperatures
    dvars = np.unique(df[densvar])
    # tvars = np.unique(df[tempvar])
    output = np.zeros((len(dvars), nx, ny))
    for idv, dv in enumerate(dvars):
        dfd = df.loc[(df[densvar]==dv)]
        amatd = amat[dfd.index]
        output[idv] = amatd.mean(axis=0)
    return output

def CLr_analysis(df, amat, is_doublon, period, dataset, center1, plot=False):
    print('CLr analysis')
    # define the mask
    stripe_peak = center1[1]-period/2
    xx, yy = array_coordinates_2d((nx, ny))
    Xmax = (10**2-(period/2)**2)**0.5-1
    beta = 5
    mask = (np.abs(yy-stripe_peak)<period/4)*1/(1+np.exp(beta*(np.abs(xx-center1[0])-Xmax)))
    ampl = -np.cos(2*np.pi*(yy-center1[1])/period)
    sign = (-1)**(xx+yy)
    weights = mask * ampl * sign
    mask1 = (np.abs(yy-stripe_peak-period)<period/4)*1/(1+np.exp(beta*(np.abs(xx-center1[0])-Xmax)))
    # ampl1 = -np.cos(2*np.pi*(yy-center1[1])/period)
    # fig, ax = plt.subplots(ncols=3)
    # dens = amat.mean(axis=0)
    # ax[0].imshow(dens)
    # ax[1].imshow(ampl*mask)
    # ax[2].imshow(dens + ampl*mask/6 + ampl1*mask1/6)
    # fig.suptitle(dataset)
    # plt.show()
    # 0/0
    # drop doublon shots, if they exist
    if is_doublon:
        df = df.loc[df[blowdetvar] != 610]
    # loop over densities/temperatures
    dvars = np.unique(df[densvar])
    # tvars = np.unique(df[tempvar])
    output = np.zeros((len(dvars), 2, nx, ny)) # second axis for nom/std
    for idv, dv in enumerate(dvars):
        dfd = df.loc[(df[densvar]==dv)]
        pp_c = {}
        for bid in range(3):
            dfb = dfd.loc[dfd[blowvar]==bid]
            amatb = amat[dfb.index]
            comp = pp_correlation_map_site_chunk(amatb, weights, nboot=500, verbose=False, print_times=False)
            comp.run(bins=None)
            pp_c[bid] = comp.A[comp.obshash['fpp_c']]
        res = 2*(pp_c[1]+pp_c[2])-pp_c[0]
        output[idv, 0] = unp.nominal_values(res)
        output[idv, 1] = unp.std_devs(res)
    output = output * sign.reshape((1,1,nx,ny))
    if plot:
        fig, ax = plt.subplots(ncols=len(dvars), nrows=2)
        if len(dvars) == 1:
            ax = ax.reshape((2,1))
        vm = np.max(np.abs(output[:,0]))
        for idv, dv in enumerate(dvars):
            ax[0,idv].imshow(output[idv,0], cmap='seismic', vmin=-vm, vmax=vm)
            ax[0,idv].set_title(dv)
            ax[1,idv].imshow(output[idv,0]*mask1, cmap='seismic', vmin=-vm, vmax=vm)
        fig.suptitle(dataset)
        plt.show()
    return output

def CLR_analysis(df, amat, is_doublon, period, dataset, center1, plot=False):
    print('CLR analysis')
    # define the mask
    stripe_peak = center1[1]-period/2
    xx, yy = array_coordinates_2d((nx, ny))
    Xmax = (10**2-(period/2)**2)**0.5-1
    beta = 5
    mask = (np.abs(yy-stripe_peak)<period/4)*1/(1+np.exp(beta*(np.abs(xx-center1[0])-Xmax)))
    ampl = -np.cos(2*np.pi*(yy-center1[1])/period)
    sign = (-1)**(xx+yy)
    weights = mask * ampl * sign
    mask1 = (np.abs(yy-stripe_peak-period)<period/4)*1/(1+np.exp(beta*(np.abs(xx-center1[0])-Xmax)))
    ampl1 = -np.cos(2*np.pi*(yy-center1[1])/period)
    weights1 = mask1 * ampl1 * sign
    weights = weights/np.sum(weights**2)**0.5
    weights1 = weights1/np.sum(weights1**2)**0.5
    if is_doublon:
        df = df.loc[df[blowdetvar] != 610]
    # loop over densities/temperatures
    dvars = np.unique(df[densvar])
    # tvars = np.unique(df[tempvar])
    output = np.zeros((len(dvars), 2, nx, ny)) # second axis for nom/std
    for idv, dv in enumerate(dvars):
        dfd = df.loc[(df[densvar]==dv)]
        pp_c = {}
        for bid in range(3):
            dfb = dfd.loc[dfd[blowvar]==bid]
            amatb = amat[dfb.index]
            comp = pp_correlation_map_chunk_chunk(amatb, weights, weights1, nboot=500, verbose=False, print_times=False)
            comp.run(bins=None)
            pp_c[bid] = comp.A[comp.obshash['fpp_c']]
        res = 2*(pp_c[1]+pp_c[2])-pp_c[0]
        output[idv, 0] = unp.nominal_values(res)
        output[idv, 1] = unp.std_devs(res)
    output = output[:,:,0,0]
    # fig, ax = plt.subplots()
    # ax.errorbar(dvars, output[:,0], yerr=output[:,1])
    # ax.set_title(dataset)
    # plt.show()
    return output

def PCA_analysis(df, amat, is_doublon, period, dataset, center1, plot=False, neig=6):
    print('PCA analysis')
    # Xmax = (10**2-(period)**2)**0.5-1
    # Xmax = int(Xmax)
    Xmax = 6
    # fig, ax = plt.subplots(ncols=2)
    # ax[0].imshow(amat.mean(axis=0))
    roi_pca = np.s_[:, center1[0]-Xmax:center1[0]+Xmax+1, center1[1]-period:center1[1]+period]
    amat = amat[roi_pca]
    # ax[1].imshow(amat.mean(axis=0))
    # fig.suptitle(dataset)
    # plt.show()
    if is_doublon:
        df = df.loc[df[blowdetvar] != 610]
    # loop over densities/temperatures
    dvars = np.unique(df[densvar])
    # tvars = np.unique(df[tempvar])
    outputv = np.zeros((len(dvars), neig, 2*Xmax+1, 2*period))
    outputw = np.zeros((len(dvars), neig))
    xx, yy = array_coordinates_2d((2*Xmax+1, 2*period))
    sign = (-1)**(xx+yy)
    sign = sign.ravel()
    for idv, dv in enumerate(dvars):
        dfd = df.loc[(df[densvar]==dv)]
        pp_c = {}
        for bid in range(3):
            dfb = dfd.loc[dfd[blowvar]==bid]
            amatb = amat[dfb.index].astype(float)
            nz = amatb.shape[0]
            m = amatb.mean(axis=0)
            dev = amatb-m.reshape((1, 2*Xmax+1, 2*period))
            dev = dev.reshape((nz, -1))
            corr = np.matmul(dev.T, dev)/nz
            corr *= (nz/(nz-1)) # bias
            corr = corr*sign.reshape((1,-1))*sign.reshape((-1,1))
            pp_c[bid] = corr
            # if bid==1:
            #     fig, ax = plt.subplots()
            #     ax.imshow(corr*4)
            #     plt.show()
        # corrmap
        # res = 2*(pp_c[1]+pp_c[2])-pp_c[0]
        res = 2*(pp_c[1]+pp_c[2])
        w, v = np.linalg.eigh(res)
        # fig, ax = plt.subplots()
        # ax.plot(w)
        # plt.show()
        # v = (v.T).reshape((res.shape[0], 2*Xmax+1, 2*period))
        w = np.flip(w)
        v = np.flip(v, axis=1)
        v = v[:,:neig]
        v = (v.T).reshape((neig, 2*Xmax+1, 2*period))
        outputv[idv, :] = v.copy()
        outputw[idv, :] = w[:neig]
    if plot:
        vm = np.max(np.abs(outputv))
        fig, ax = plt.subplots(ncols=len(dvars), nrows=neig)
        if len(dvars) == 1:
            ax = ax.reshape((neig, 1))
        for idv, dv in enumerate(dvars):
            for ie in range(neig):
                ax[ie, idv].imshow(outputv[idv, ie, :,:], cmap='seismic', vmin=-vm, vmax=vm)
            ax[0,idv].set_title(dv)
        fig.suptitle(dataset+' pca')
        plt.show()
    return outputw, outputv

def singles_avg_analysis(df, amat, is_doublon, period, center1):
    print('singles avg analysis')
    Xmax = 6
    roi_avg = np.s_[:, center1[0]-Xmax:center1[0]+Xmax+1, center1[1]-period:center1[1]+period]
    amat = amat[roi_avg]
    amat = amat.mean(axis=(1,2))
    if is_doublon:
        df = df.loc[df[blowdetvar] != 610]
    # loop over densities/temperatures
    dvars = np.unique(df[densvar])
    # tvars = np.unique(df[tempvar])
    output = np.zeros((len(dvars), 3, 2))
    for idv, dv in enumerate(dvars):
        dfd = df.loc[(df[densvar]==dv)]
        for bid in range(3):
            dfb = dfd.loc[dfd[blowvar]==bid]
            amatb = amat[dfb.index].astype(float)
            m = amatb.mean()
            s = amatb.std()/len(amatb)**0.5
            output[idv, bid, 0] = m
            output[idv, bid, 1] = s
    uout = unp.uarray(output[:,:,0], output[:,:,1])
    uout = p_avg_weight*uout[:,0]+(1-p_avg_weight)*(uout[:,1]+uout[:,2])
    output = np.array([unp.nominal_values(uout), unp.std_devs(uout)]).T
    return output

def doublons_avg_analysis(df, amat, period, center1):
    print('doublon avg analysis')
    Xmax = 6
    roi_avg = np.s_[:, center1[0]-Xmax:center1[0]+Xmax+1, center1[1]-period:center1[1]+period]
    amat = amat[roi_avg]
    amat = amat.mean(axis=(1,2))
    df = df.loc[df[blowdetvar] == 610]
    # loop over densities/temperatures
    dvars = np.unique(df[densvar])
    # tvars = np.unique(df[tempvar])
    output = np.zeros((len(dvars), 2))
    for idv, dv in enumerate(dvars):
        dfd = df.loc[(df[densvar]==dv)]
        amatb = amat[dfd.index].astype(float)
        m = amatb.mean()
        s = amatb.std()/len(amatb)**0.5
        output[idv, 0] = m
        output[idv, 1] = s
    return output

def inspect_group(df, amat, is_doublon):
    if is_doublon:
        df = df.loc[df[blowdetvar] != 610]
    dfs = {bid: df.loc[df[blowvar] == bid] for bid in range(3)}
    amats = {bid: amat[dfs[bid].index] for bid in range(3)}
    atnums = {bid: amats[bid].sum(axis=(1,2)) for bid in range(3)}
    fig, ax = plt.subplots()
    for bid in range(3):
        ax.plot(atnums[bid], label=bid)
    ax.legend()
    plt.show()

def analyze_scan_group(scan_group, dataset):
    all_data = {} # object in which we put data
    is_doublon = scan_group['is_doublon_resolved']
    period = scan_group['period']
    shift_center = scan_group['shift_center']
    center1 = [center[0]+shift_center[0], center[1]+shift_center[1]]
    # load data
    scanstrings = scan_group['data']
    if is_doublon: # handle specially because of dmd issue
        df, amat = load_processed_scans(scanstrings, roi_hwidth=45, yx0=yxCentersite, dmd_fixer=False, runlog_path=runlog, load_direc=load_dir)
        amat = amat[:, AACrop[0]:AACrop[1], AACrop[2]:AACrop[3]]
        df = ad_hoc_postselection(df)
        amat = amat[df.index]
        df = df.reset_index(drop=True)
        doublon_fidelity = get_doublon_fidelity(df, amat, center1, dataset)
        all_data['doublon_fidelity'] = doublon_fidelity
    df, amat = load_processed_scans(scanstrings, roi_hwidth=45, yx0=yxCentersite, dmd_fixer=True, runlog_path=runlog, load_direc=load_dir)
    # our usual cropping
    amat = amat[:, AACrop[0]:AACrop[1], AACrop[2]:AACrop[3]]
    # global postselection
    df = ad_hoc_postselection(df)
    amat = amat[df.index]
    df = df.reset_index(drop=True)
    # doublon calibration
    if is_doublon:
        df, amat = drop_doublon_calibration(df, amat)
    # # atom number postselection
    df, amat = atom_number_postselection(df, amat, is_doublon)
    # # inspect
    # inspect_group(df, amat, is_doublon)
    # singles analysis
    dvars, singles_data = singles_map_analysis(df, amat, is_doublon)
    all_data['singles_map'] = singles_data
    all_data['densvar'] = dvars
    singles_avg = singles_avg_analysis(df, amat, is_doublon, period, center1)
    all_data['singles_avg'] = singles_avg
    # doublon
    if is_doublon:
        doublon_data = doublon_map_analysis(df, amat, is_doublon)
        all_data['doublon_map'] = doublon_data
        doublon_avg = doublons_avg_analysis(df, amat, period, center1)
        all_data['doublon_avg'] = doublon_avg
    # CLr analysis
    CLr = CLr_analysis(df, amat, is_doublon, period, dataset, center1)
    all_data['CLr'] = CLr
    # # CLR analysis
    CLR = CLR_analysis(df, amat, is_doublon, period, dataset, center1)
    all_data['CLR'] = CLR
    # PCA analysis
    PCAw, PCAv = PCA_analysis(df, amat, is_doublon, period, dataset, center1)
    all_data['PCAw'] = PCAw
    all_data['PCAv'] = PCAv
    return all_data

# todo
# singles and doublons, averaged in the ROI
# start making plots

for dataset in scan_groups:
    print('analyzing {}'.format(dataset))
    processed_data = analyze_scan_group(scan_groups[dataset], dataset)
    # save stuff
    singles_map = processed_data['singles_map']
    dvars = processed_data['densvar']
    np.save('processed/{}_singles_map'.format(dataset), singles_map)
    np.save('processed/{}_densvar'.format(dataset), dvars)
    singles_avg = processed_data['singles_avg']
    np.save('processed/{}_singles_avg'.format(dataset), singles_avg)
    if scan_groups[dataset]['is_doublon_resolved']:
        doublon_map = processed_data['doublon_map']
        np.save('processed/{}_doublon_map'.format(dataset), doublon_map)
        doublon_avg = processed_data['doublon_avg']
        np.save('processed/{}_doublon_avg'.format(dataset), doublon_avg)
        doublon_fidelity = processed_data['doublon_fidelity']
        np.save('processed/{}_doublon_fidelity'.format(dataset), np.array([doublon_fidelity.n, doublon_fidelity.s]))
    CLr = processed_data['CLr']
    np.save('processed/{}_CLr'.format(dataset), CLr)
    CLR = processed_data['CLR']
    np.save('processed/{}_CLR'.format(dataset), CLR)
    PCAw = processed_data['PCAw']
    PCAv = processed_data['PCAv']
    np.save('processed/{}_PCAw'.format(dataset), PCAw)
    np.save('processed/{}_PCAv'.format(dataset), PCAv)
plt.show()



