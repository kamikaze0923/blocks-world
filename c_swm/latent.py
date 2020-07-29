import argparse
import torch
import utils
import os
import pickle
from c_swm.plot import TransitionPlot

from torch.utils import data
import numpy as np

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

dataset = utils.StateTransitionsDataset(hdf5_file=args.dataset, action_encoding=args.action_encoding, truncate=50)
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
    encoder=args.encoder,
    action_encoding=args.action_encoding
).to(device)

model.load_state_dict(torch.load(model_file,  map_location=device))
model.eval()

with torch.no_grad():
    tr_plot = TransitionPlot()

    for batch_idx, data_batch in enumerate(eval_loader):
        data_batch = [tensor.to(device) for tensor in data_batch]
        tr_plot.reset()

        obs = data_batch[0]
        action = data_batch[1]
        next_obs = data_batch[-1]
        tr_plot.plt_observations(obs, next_obs)

        if args.action_encoding == "action_image":
            tr_plot.plt_action(action)

        objs = model.obj_extractor(obs)
        next_objs = model.obj_extractor(next_obs)
        tr_plot.plt_objects(objs, next_objs)

        state = model.obj_encoder(objs)
        next_state = model.obj_encoder(next_objs)

        pred_trans = model.transition_model(state, action)
        pred_state = state + pred_trans

        tr_plot.plt_latent(state, next_state, pred_state)
        tr_plot.show()

    tr_plot.close()


