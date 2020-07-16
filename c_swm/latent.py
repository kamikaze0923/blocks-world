import argparse
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

import c_swm.modules as modules

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

args_eval = parser.parse_args()


meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')

args = pickle.load(open(meta_file, 'rb'))['args']

args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 100
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

dataset = utils.StateTransitionsDataset(hdf5_file=args.dataset)
eval_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Get data sample
obs = eval_loader.__iter__().next()[0]
input_shape = obs[0].size()

model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    encoder=args.encoder).to(device)

model.load_state_dict(torch.load(model_file,  map_location='cpu'))
model.eval()


with torch.no_grad():
    fig, axs = plt.subplots(2, 3, figsize=(10,6))
    axs[0,0].axis('off')
    axs[0,1].axis('off')
    axs[0,2].axis('off')
    colors = ['y', 'r', 'g', 'b']
    for batch_idx, data_batch in enumerate(eval_loader):

        obs = data_batch[0]
        action = data_batch[1]
        next_obs = data_batch[-1]

        state = model.obj_encoder(model.obj_extractor(obs))
        next_state = model.obj_encoder(model.obj_extractor(next_obs))

        pred_trans = model.transition_model(state, action)
        pred_state = state + pred_trans


        n_obj = args.num_objects
        assert n_obj == state.size()[1]

        np_obs = np.transpose(obs[0].cpu().numpy(), (1,2,0))
        np_next_obs = np.transpose(next_obs[0].cpu().numpy(), (1,2,0))
        print(np_obs.shape, np_next_obs.shape)
        axs[0,0].imshow(np_obs)
        axs[0,0].set_title("Pre State", fontsize=8)
        axs[0,1].imshow(np_next_obs)
        axs[0,1].set_title("Next State", fontsize=8)
        axs[0,2].imshow(np_next_obs)
        axs[0,2].set_title("Next State", fontsize=8)
        axs[1,0].cla()
        axs[1,1].cla()
        axs[1,2].cla()
        axs[1,0].set_title("Pre State Latent", fontsize=8)
        axs[1,1].set_title("Next State Latent", fontsize=8)
        axs[1,2].set_title("Pre State Latent +\n Transition", fontsize=8)
        axs[1,0].set_xlim(-15, 15)
        axs[1,0].set_ylim(-5, 15)
        axs[1,1].set_xlim(-15, 15)
        axs[1,1].set_ylim(-5, 15)
        axs[1,2].set_xlim(-15, 15)
        axs[1,2].set_ylim(-5, 15)

        for i in range(n_obj):
            np_state = state[0][i].cpu().numpy()
            np_next_state = next_state[0][i].cpu().numpy()
            np_pred_state = pred_state[0][i].cpu().numpy()
            if i == 0:
                print(np_state, np_next_state, np_pred_state)
                print("-*40")

            axs[1,0].scatter(np_state[0], np_state[1], color=colors[i], marker='x', s=10)
            axs[1,1].scatter(np_next_state[0], np_next_state[1], color=colors[i], marker='x', s=10)
            axs[1,2].scatter(np_pred_state[0], np_pred_state[1], color=colors[i], marker='x', s=10)


        axs[1,0].legend(['Object {}'.format(i) for i,_ in enumerate(colors)], prop={'size': 6}, loc=2, ncol=2)
        axs[1,1].legend(['Object {}'.format(i) for i,_ in enumerate(colors)], prop={'size': 6}, loc=2, ncol=2)
        axs[1,2].legend(['Object {}'.format(i) for i,_ in enumerate(colors)], prop={'size': 6}, loc=2, ncol=2)
        plt.pause(0.5)
    plt.close()


