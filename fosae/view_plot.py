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

action = np.load("fosae/block_data/action.npy")
print(action)
exit(0)


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

        for i, (p, p_nt, a) in enumerate(zip(one_p, one_p_nt, one_a)):
            axs[2,i].imshow(p, cmap='gray')
            axs[3,i].imshow(p_nt, cmap='gray')
            axs[4,i].imshow(a, cmap='gray')

        plt.pause(0.2)
        # a = 1



