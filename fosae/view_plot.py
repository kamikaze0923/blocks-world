import matplotlib.pyplot as plt
import numpy as np
from fosae.modules import P

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
print(action.shape, preds.shape, preds_next.shape)



fig, axs = plt.subplots(8, 9, figsize=(8, 6))
for _, ax in np.ndenumerate(axs):
    ax.set_xticks([])
    ax.set_yticks([])
plt.gca()


while True:
    for one_data, one_rec_batch, one_preds, one_args, one_data_next, one_rec_batch_next, one_preds_next, one_args_next\
            in zip(data, rec_batch, preds, args, data_next, rec_batch_next, preds_next, args_next):
        for i, (d, r, p, ars, d_nt, r_nt, p_nt, ars_nt) in enumerate(
            zip(one_data, one_rec_batch, one_preds, one_args, one_data_next, one_rec_batch_next, one_preds_next, one_args_next)
        ):


            axs[0,i].imshow(np.transpose(r, (1,2,0)))
            axs[1,i].imshow(np.transpose(r_nt, (1,2,0)))

            axs[2,i].imshow(p[:18], cmap='gray')
            axs[3,i].imshow(p_nt[:18], cmap='gray')

            axs[4,i].imshow(p[18:], cmap='gray')
            axs[5,i].imshow(p_nt[18:], cmap='gray')

            axs[6,i].imshow(np.abs(p - p_nt)[:18], cmap='gray')
            axs[7, i].imshow(np.abs(p - p_nt)[18:], cmap='gray')




        plt.pause(0.2)



