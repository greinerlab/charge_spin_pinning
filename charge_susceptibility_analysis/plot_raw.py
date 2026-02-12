import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

with open('scan_group_info.yaml', "r") as file:
    scan_groups = yaml.safe_load(file)

for dataset in scan_groups:
    # if dataset == "U8_Oct_9_p_58":
    filename=f"processed/{dataset}_output2d.npz"
    if os.path.exists(filename):
        print(dataset)
        arr = np.load(filename, allow_pickle=True)

        print(list(arr.keys()))
        periods = arr['periods']
        idxs = np.argsort(periods)
        # print(periods)
        # print(arr['diff_doublons'].shape)
        fig, ax = plt.subplots(nrows=3, figsize=(5, 6))
        fig.suptitle(dataset)
        im=ax[0].imshow(arr['diff_singles'][idxs], origin='lower', cmap='bwr', vmin=-0.05, vmax=0.05)
        ax[0].set_yticks(range(len(periods)))
        ax[0].set_yticklabels(periods[idxs])
        ax[0].set_title('singles')
        fig.colorbar(im)
        im=ax[1].imshow(arr['diff_doublons'][idxs], origin='lower', cmap='bwr', vmin=-0.05, vmax=0.05)
        ax[1].set_yticks(range(len(periods)))
        ax[1].set_yticklabels(periods[idxs])
        fig.colorbar(im)
        ax[1].set_title('doublons')

        im=ax[2].imshow(arr['diff_dens'][idxs], origin='lower', cmap='bwr', vmin=-0.05, vmax=0.05)
        ax[2].set_yticks(range(len(periods)))
        ax[2].set_yticklabels(periods[idxs])
        fig.colorbar(im)
        ax[2].set_title('density')

plt.show()