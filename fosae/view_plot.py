import matplotlib.pyplot as plt
import numpy as np

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



fig, axs = plt.subplots(4, 9, figsize=(12,12))
for _, ax in np.ndenumerate(axs):
    ax.axis('off')
plt.gca()

while True:
    for one_data, one_rec_batch, one_preds, one_args, one_data_next, one_rec_batch_next, one_preds_next, one_args_next\
            in zip(data, rec_batch, preds, args, data_next, rec_batch_next, preds_next, args_next):
        for i, (d, r, p0, p1, ars0, ars1, d_nt, r_nt, p_nt0, p_nt1, ars_nt) in enumerate(
            zip(one_data, one_rec_batch, one_preds[:9], one_preds[9:], one_args[:9], one_args[9:], one_data_next, one_rec_batch_next, one_preds_next[:9], one_preds_next[9:], one_args_next)
        ):
            axs[0,i].imshow(np.transpose(d, (1,2,0)))
            axs[1,i].imshow(np.transpose(r, (1,2,0)))
            # axs[2,i].imshow(p0, cmap='gray')
            # axs[3,i].imshow(p1, cmap='gray')
            # axs[4+j,i].imshow(np.transpose(ars, (1,2,0)))

            axs[2,i].imshow(np.transpose(d_nt, (1,2,0)))
            axs[3,i].imshow(np.transpose(r_nt, (1,2,0)))
            # axs[6,i].imshow(p_nt0, cmap='gray')
            # axs[7,i].imshow(p_nt1, cmap='gray')


        plt.pause(0.01)



