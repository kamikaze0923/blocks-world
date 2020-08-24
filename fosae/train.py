from c_swm.utils import StateTransitionsDataset
from torch.utils.data import DataLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = StateTransitionsDataset(hdf5_file="c_swm/data/blocks_train.h5", n_obj=9)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

N_EPOCH = 20


for epoch in range(1, N_EPOCH + 1):

    train_loss = 0

    for batch_idx, data_batch in enumerate(train_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]

        obs, _, _, _, _, _, _ = data_batch
        print(obs.size())
        exit(0)




    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.6f}'.format(
        epoch, avg_loss))

    if avg_loss < best_loss:
        best_loss = avg_loss



