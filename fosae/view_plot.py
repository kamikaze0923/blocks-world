import matplotlib.pyplot as plt
import numpy as np
from fosae.get_view_data import MAX_N

def dis(preds, preds_next):
    for a,b in zip(preds, preds_next):
        print(a-b)
        print("-"*10)

data = np.load("fosae/block_data/block_data.npy")
preds = np.load("fosae/block_data/block_preds.npy")

data_next = np.load("fosae/block_data/block_data_next.npy")
preds_next = np.load("fosae/block_data/block_preds_next.npy")

action = np.load("fosae/block_data/change.npy")

fig, axs = plt.subplots(5, MAX_N, figsize=(8, 6))
for _, ax in np.ndenumerate(axs):
    ax.set_xticks([])
    ax.set_yticks([])
plt.gca()

def show_img(ax, arr):
    ax.imshow(np.transpose(arr, (1,2,0)))

while True:
    for one_data, one_data_next, one_p, one_p_nt, one_a in zip(
        data, data_next, preds, preds_next, action
    ):
        for i, (d, d_nt) in enumerate(zip(one_data, one_data_next)):
            show_img(axs[0,i], d)
            show_img(axs[1,i], d_nt)


        axs[2,0].imshow(one_p, cmap='gray')
        axs[3,0].imshow(one_p_nt, cmap='gray')
        axs[4,0].imshow(one_a, cmap='gray')
        print(np.abs(0.5-one_p) > 0.49)
        print("-"*20)
        print(np.abs(0.5-one_p_nt) > 0.49)
        print("-"*20)
        print(one_a)
        print("-"*20)

        plt.pause(0.2)
        # a = 1



