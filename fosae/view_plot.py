import matplotlib.pyplot as plt
import numpy as np
from fosae.modules import A





data = np.load("fosae/block_data/block_data_fo.npy")
rec_batch = np.load("fosae/block_data/block_rec_fo.npy")
args = np.load("fosae/block_data/block_args_fo.npy")
preds = np.load("fosae/block_data/block_preds_fo.npy")

fig, axs = plt.subplots(4 + args.shape[2], 9, figsize=(6,6))
for _, ax in np.ndenumerate(axs):
    ax.axis('off')
plt.gca()

while True:
    for one_data, one_rec_batch, one_preds, one_args in zip(data, rec_batch, preds, args):
        for i, (d, r, p, ars) in enumerate(zip(one_data, one_rec_batch, one_preds, one_args)):
            axs[0,i].imshow(np.transpose(d, (1,2,0)))
            axs[1,i].imshow(np.transpose(r, (1,2,0)))
            axs[2,i].imshow(p, cmap='gray')
            for j, ars in enumerate(one_args[i]):
                axs[3+j,i].imshow(np.transpose(ars, (1,2,0)))

        plt.pause(0.1)



