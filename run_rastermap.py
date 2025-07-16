from pathlib import Path
import json
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap
from tqdm import tqdm
from scipy.stats import zscore

root = "C://Users//Josh Selfe//OneDrive - Nexus365//Other Documents//Neuromatch"

sup_bef = 'VR2_2021_03_20_1' #example mouse before supervised learning
sup_aft = 'VR2_2021_04_06_1' #example mouse after supervised learning
unsup_bef = 'TX105_2022_10_08_2' #example mouse before supervised learning
unsup_aft = 'TX105_2022_10_19_2' #example mouse after supervised learning

svd_dec_400pc = np.load(os.path.join(root, sup_aft +'_SVD_dec.npy'), allow_pickle=1).item() # 400 PCs
spks = svd_dec_400pc['U'][:, :].T @ svd_dec_400pc['V'] # project from the PC space back to neural space (only do it for the first 1000 neuorns)
spks = zscore(spks, axis=1)
nfrs = spks.shape[1]
subset = spks[:1000, :]

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.plot(spks[i], label=f'Neuron {i+1}')

plt.xlabel("Time (or frame index)")
plt.ylabel("Fluorescence")
plt.title("Fluorescence traces of first 10 neurons")
plt.legend()
plt.tight_layout()
plt.show()

#%%
model = Rastermap(n_PCs=200, n_clusters=100, 
                  locality=0.75, time_lag_window=5).fit(spks)

y = model.embedding # neurons x 1
isort = model.isort

# visualize binning over neurons
X_embedding = model.X_embedding

# plot
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
plt.show()
