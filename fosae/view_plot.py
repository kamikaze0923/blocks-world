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



fig, axs = plt.subplots(4, 9, figsize=(18, 8))
for _, ax in np.ndenumerate(axs):
    ax.set_xticks([])
    ax.set_yticks([])
plt.gca()

MARGIN = np.zeros(shape=(P, 8))

while True:
    for one_data, one_rec_batch, one_preds, one_args, one_data_next, one_rec_batch_next, one_preds_next, one_args_next\
            in zip(data, rec_batch, preds, args, data_next, rec_batch_next, preds_next, args_next):
        for i, (d, r, p0, p1, ars0, ars1, d_nt, r_nt, p_nt0, p_nt1, ars_nt0, ars_nt1) in enumerate(
            zip(one_data, one_rec_batch, one_preds[:9], one_preds[9:], one_args[:9], one_args[9:],
                one_data_next, one_rec_batch_next, one_preds_next[:9], one_preds_next[9:], one_args_next[9:], one_args_next[:9])
        ):
            axs[0,i].imshow(np.transpose(d, (1,2,0)))
            axs[1,i].imshow(np.transpose(d_nt, (1,2,0)))

            axs[2,i].imshow(np.transpose(r, (1,2,0)))
            axs[3,i].imshow(np.transpose(r_nt, (1,2,0)))

            # axs[4,i].imshow(np.concatenate([p0, MARGIN, p1], axis=1), cmap='gray')
            # axs[5,i].imshow(np.concatenate([p_nt0, MARGIN, p_nt1], axis=1), cmap='gray')
            #
            # axs[6,i].imshow(np.concatenate([np.abs(p0 - p_nt0), MARGIN, np.abs(p1 - p_nt1)], axis=1), cmap='gray')

            # axs[7,i].imshow(np.transpose(ars0[0], (1,2,0)))
            # axs[8,i].imshow(np.transpose(ars0[1], (1,2,0)))
            # axs[9,i].imshow(np.transpose(ars1[0], (1,2,0)))
            # axs[10,i].imshow(np.transpose(ars1[1], (1,2,0)))


        plt.pause(0.1)



