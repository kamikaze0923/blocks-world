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
preds = np.load("fosae/block_data/block_preds.npy")

data_next = np.load("fosae/block_data/block_data_next.npy")
rec_batch_next = np.load("fosae/block_data/block_rec_next.npy")
preds_next = np.load("fosae/block_data/block_preds_next.npy")

action = np.load("fosae/block_data/action.npy")


fig, axs = plt.subplots(7, N, figsize=(8, 6))
for _, ax in np.ndenumerate(axs):
    ax.set_xticks([])
    ax.set_yticks([])
plt.gca()

def show_img(ax, arr):
    ax.imshow(np.transpose(arr, (1,2,0)))


while True:
    for one_data, one_rec_batch, one_data_next, one_rec_batch_next, one_p, one_p_nt, one_a in zip(
        data, rec_batch, data_next, rec_batch_next, preds, preds_next, action
    ):
        for i, (d, r, d_nt, r_nt) in enumerate(zip(one_data, one_rec_batch, one_data_next, one_rec_batch_next)):
            show_img(axs[0,i], r)
            show_img(axs[1,i], r_nt)
            show_img(axs[2,i], d)
            show_img(axs[3,i], d_nt)


        for i, (p, p_nt, a) in enumerate(zip(one_p, one_p_nt, one_a)):
            axs[4,i].imshow(p, cmap='gray')
            axs[5,i].imshow(p_nt, cmap='gray')
            axs[6,i].imshow(a, cmap='gray')



        plt.pause(0.2)
        # a = 1



