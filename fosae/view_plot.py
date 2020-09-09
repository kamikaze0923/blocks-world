import matplotlib.pyplot as plt
import numpy as np
import pickle
from fosae.modules import IMG_H, IMG_W, IMG_C, A, N, U, P


def dis(preds, preds_next):
    for a,b in zip(preds, preds_next):
        print(a-b)
        print("-"*10)



data = np.load("fosae/block_data/block_data.npy")
rec_batch = np.load("fosae/block_data/block_rec.npy")
args = np.load("fosae/block_data/block_args.npy")
preds = np.load("fosae/block_data/block_preds.npy")

data_next = np.load("fosae/block_data/block_data_next.npy")
rec_batch_next = np.load("fosae/block_data/block_rec_next.npy")
args_next = np.load("fosae/block_data/block_args_next.npy")
preds_next = np.load("fosae/block_data/block_preds_next.npy")

action = np.load("fosae/block_data/block_action.npy")

fig, axs = plt.subplots(6, N, figsize=(8, 6))
for _, ax in np.ndenumerate(axs):
    ax.set_xticks([])
    ax.set_yticks([])
plt.gca()


while True:
    for one_data, one_rec_batch, one_data_next, one_rec_batch_next, one_p, one_p_nt in zip(
        data, rec_batch, data_next, rec_batch_next, preds, preds_next
    ):
        for i, (d, r, d_nt, r_nt) in enumerate(zip(one_data, one_rec_batch, one_data_next, one_rec_batch_next)):

            axs[0,i].imshow(np.transpose(r, (1,2,0)))
            axs[1,i].imshow(np.transpose(r_nt, (1,2,0)))

            axs[2,i].imshow(np.transpose(d, (1,2,0)))
            axs[3,i].imshow(np.transpose(d_nt, (1,2,0)))



            # axs[6,i].imshow(p_a, cmap='gray')
            #one_p
            #
            # axs[7,i].imshow(np.abs(p_nt - p_a), cmap='gray')
            #
            # axs[8,i].imshow(np.transpose(ars[0], (1,2,0)))
            # axs[9,i].imshow(np.transpose(ars[1], (1,2,0)))
            # axs[10,i].imshow(np.transpose(ars_nt[0], (1,2,0)))
            # axs[11,i].imshow(np.transpose(ars_nt[1], (1,2,0)))

        axs[4,0].imshow(one_p[0], cmap='gray')
        axs[5,0].imshow(one_p_nt[0], cmap='gray')
        plt.pause(0.2)



