import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from nets.resnet import ConvNeXt, InvResX1D
from dataset.vox_dataset import VoxDataset
from dataset.utils import collate_vox_lips
from argparse import ArgumentParser

from torch.utils.tensorboard import SummaryWriter
import pickle

# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'

assert torch.cuda.is_available()
device = 'cuda'

n_pts_coord = 5
n_pts_melspec = 20
N = 48

# CNN layer structure
layers = {
    '0': [2, 2, 2, 2],
    '1': [3, 4, 6, 3]
}
# Spectro dimension under two different preprocessings
a_dim = {
    '1': 26
}
# CNN architecture
archis = {
    'conv1d_x': ConvNeXt,
    'conv1d_a': ConvNeXt
}

class LipSyncNet(nn.Module):

    def __init__(self, args, params):
        super(LipSyncNet, self).__init__()

        self.args = args
        in_dims = 20

        self.emb = nn.Linear(in_dims, 272)

        params_x, params_a = params
        self.encoder_x, self.encoder_a = archis[self.args.e_x](**params_x), archis[self.args.e_a](**params_a)


    def forward(self, x, a, norm=True):
        '''
        x shape: N, n_pts_coord, 20
        a shape: N, n_pts_melspec, a_dim[str(args.audio_style)]
        '''

        e_x = self.emb(x.flatten(start_dim=-2))

        ## Landmarks encoding
        e_x = e_x.transpose(1, 2)
        e_x = self.encoder_x(e_x)
        if norm:
            e_x = e_x / e_x.norm(dim=-1, keepdim=True)
        ## Spectro encoding
        e_a = a.transpose(1, 2)
        e_a = self.encoder_a(e_a)
        if norm:
            e_a = e_a / e_a.norm(dim=-1, keepdim=True)

        return e_x, e_a


def prepare_batch(batch, syncnet_style=False, n_splits=12):

    motion_feat, mel, lengths = batch
    splits = []
    idx_x, idx_a = [np.cumsum([0] + list(l)) for l in zip(*lengths)]
    increments = [l_x / l_a for l_x, l_a in lengths]

    for i_x0, i_xf, i_a0, i_af, inc in zip(idx_x[:-1], idx_x[1:], idx_a[:-1], idx_a[1:], increments):
        m = mel[i_a0:i_af]
        x = motion_feat[i_x0:i_xf]
        off = np.random.randint(n_pts_melspec)
        i_a = np.arange(off, len(m) - n_pts_melspec, n_pts_melspec)
        i_x = np.round(i_a * inc).astype(int)
        if syncnet_style: ## In this case the negative pairs are chosen from the same sample, all n_splits pairs in a sample form a batch
            splits.append([(x[i:i + n_pts_coord], m[j:j + n_pts_melspec]) for (i, j) in zip(i_x, i_a)])
        else:
            splits.extend([(x[i:i + n_pts_coord], m[j:j + n_pts_melspec]) for (i, j) in zip(i_x, i_a)])

    if syncnet_style:
        x, a = torch.Tensor(0), torch.Tensor(0)
        for sample_splits in splits:
            if len(sample_splits) >= n_splits:
                rg = np.arange(len(sample_splits))
                np.random.shuffle(rg)
                split_idx = np.split(rg[:n_splits * (len(rg) // n_splits)], len(rg) // n_splits)
                for indices in split_idx:
                    samples = [sample_splits[i] for i in indices]
                    x_split, a_split = [torch.stack(s).unsqueeze(0) for s in zip(*samples)]
                    x, a = torch.cat([x, x_split]), torch.cat([a, a_split])
        return x, a

    if len(splits) < N:
        return None, None

    # Allow to sample several combinations of N samples to save I/O time
    rg = np.arange(len(splits))
    np.random.shuffle(rg)
    split_idx = np.split(rg[:N * (len(rg) // N)], len(rg) // N)
    x, a = torch.Tensor(0), torch.Tensor(0)
    for indices in split_idx:
        samples = [splits[i] for i in indices]
        x_split, a_split = [torch.stack(s).unsqueeze(0) for s in zip(*samples)]
        x, a = torch.cat([x, x_split]), torch.cat([a, a_split])

    return x, a


def nce_loss(e_a, e_x, delta, loss_type, n_splits, margin=10):

    if loss_type == 'syncnet':
        pos_d = ((e_x - e_a) ** 2).sum(dim=-1)
        neg_d = ((e_x - torch.roll(e_a, 1, 0)) ** 2).sum(dim=-1)
        neg_d = torch.clamp(margin - neg_d ** 0.5, min=0) ** 2
        return 0.5 * (pos_d.mean() + neg_d.mean())


    dot_pdt = torch.matmul(e_a, e_x.transpose(0, 1)) - torch.diag(delta * torch.ones(n_splits)).to(device)

    if loss_type == 'bce':
        ### Sigmoid
        sigmo_out = 1 / (1 + torch.exp(-dot_pdt))
        loss = -0.5 * (2 * torch.log(torch.diag(sigmo_out)[:-1]).mean() + torch.log(1 - torch.diag(sigmo_out, 1)).mean() \
            + torch.log(1 - torch.diag(sigmo_out, -1)).mean())

    elif loss_type == 'infonce':
        dot_pdt = torch.exp(dot_pdt)
        pos = torch.diag(dot_pdt)
        l_xa = pos / dot_pdt.sum(dim=0)
        l_xa = -torch.log(l_xa)
        l_ax = pos / dot_pdt.sum(dim=1)
        l_ax = -torch.log(l_ax)
        
        loss = l_xa.mean() + l_ax.mean()
    
    return loss


def train(model, loader, val_loader, optimizer, lr_scheduler, writer, args):

    out_dir = os.path.join('models', args.out_dir)
    if os.path.exists(os.path.join(out_dir, f'model_chkpt.pt')):
        # Resume
        save_dict = torch.load(os.path.join(out_dir, 'model_chkpt.pt'))
        model.load_state_dict(save_dict['checkpoints'])
        optimizer.load_state_dict(save_dict['optimizer'])
        lr_scheduler.load_state_dict(save_dict['scheduler'])
        loader = save_dict['loader']
        val_loader = save_dict['val_loader']
        epoch = save_dict['epoch']
        steps = save_dict['steps']
        delta = save_dict['delta']
    else:
        steps = 0
        epoch = 0
        delta = args.delta
    loss_type = args.loss
    syncnet_style = args.syncnet_style # and (loss_type == 'syncnet')
    

    # save args
    with open(os.path.join(out_dir, 'args'), 'wb') as f:
        pickle.dump(args, f)

    while steps < args.steps:

        for batch in loader:
            
            b_x, b_a = prepare_batch(batch, syncnet_style=syncnet_style)
            if b_x is None:
                continue
            b_x, b_a = b_x.to(device), b_a.to(device)

            for b_idx in range(len(b_x)):
                x, a = b_x[b_idx], b_a[b_idx]

                optimizer.zero_grad()
                e_a, e_x = model(x, a, norm=(loss_type != 'syncnet'))
                loss = nce_loss(e_a, e_x, delta, loss_type, 12 if syncnet_style else N)
                loss.backward()
                optimizer.step()

                if steps % 1000 == 0:
                    if steps > 0:
                        delta = min(delta * args.delta_gamma, 0.4)
                        lr_scheduler.step()
                    lr = lr_scheduler.get_last_lr()[0]
                    writer.add_scalar('Lr', lr, global_step=steps)
                    writer.add_scalar('Delta', delta, global_step=steps)
                    writer.add_scalar('Loss/Train', loss.item(), global_step=steps)
                    model.eval()
                    val_loss = 0
                    i = 0
                    for val_batch in val_loader:
                        b_x_val, b_a_val = prepare_batch(val_batch, syncnet_style=syncnet_style)
                        if b_x_val is None:
                            continue
                        b_x_val, b_a_val = b_x_val.to(device), b_a_val.to(device)
                        for b_idx_val in range(len(b_x_val)):
                            x, a = b_x_val[b_idx_val], b_a_val[b_idx_val]
                            with torch.no_grad():
                                e_a, e_x = model(x, a, norm=(loss_type != 'syncnet'))
                            val_loss += nce_loss(e_a, e_x, 0, loss_type, 12 if syncnet_style else N).item()
                            i += 1
                    if i > 0:
                        writer.add_scalar('Loss/Val', val_loss / i, global_step=steps)
                    model.train()
                    save_dict = dict(
                        checkpoints=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=lr_scheduler.state_dict(),
                        loader=loader,
                        val_loader=val_loader,
                        steps=steps,
                        epoch=epoch,
                        delta=delta
                    )
                    torch.save(save_dict, os.path.join(out_dir, f'model_chkpt.pt'))
                    print(f'Step {steps} done, loss: {loss.item()}, delta: {delta}')
                    
                steps += 1
        epoch += 1

    save_dict = dict(
        checkpoints=model.state_dict()
    )
    torch.save(save_dict, os.path.join(out_dir, f'model_chkpt_last.pt'))
    print(f'Training done, loss: {loss.item()}, delta: {delta}')


def main(args):

    dataset = VoxDataset(args.data_dir, args.audio_dir, pyramid_level=args.pyramid_level, 
                         kernel_size=args.pyramid_kernel_size)
    train_set, val_set = random_split(dataset, [int(len(dataset) * 0.98), len(dataset) - int(len(dataset) * 0.98)])
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_vox_lips, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_vox_lips, pin_memory=True)

    ## tensorboard
    out_dir = os.path.join('results', args.out_dir)
    tb_dir = os.path.join(out_dir, 'log')
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    parameters = {
            'conv1d_x': dict(inplanes=272, block=InvResX1D, layers=layers[str(args.conv_layers)], out_dim=args.e_dim, sa=args.sa_x),
            'conv1d_a': dict(inplanes=a_dim[str(args.audio_style)], block=InvResX1D, layers=layers[str(args.conv_layers)], out_dim=args.e_dim, sa=args.sa_a)
        }
    params = parameters[args.e_x], parameters[args.e_a]

    model = LipSyncNet(args, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.step_lrscheduler:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    else:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

    train(model, loader, val_loader, optimizer, lr_scheduler, writer, args)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--data_dir', help='Path to motion features')
    parser.add_argument('--audio_dir', help='Path to audio features')
    parser.add_argument('--out_dir', help='Path to output directory')
    # Learning params
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument('--step_lrscheduler', action='store_true', help='step vs exponential LR')
    parser.add_argument('--lr_gamma', type=float, default=0.999)
    parser.add_argument('--step_size', type=float, default=1000)
    parser.add_argument('--delta', type=float, default=0, help='InfoNCE initial margin, 0 means no change during training')
    parser.add_argument('--delta_gamma', type=float, default=1.002)
    parser.add_argument('--steps', type=int, default=3e6) # eq to ~500 epochs
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--loss', type=str, choices=['bce', 'infonce', 'syncnet'], default='bce', help='Which type of loss')
    parser.add_argument('--syncnet_style', action='store_true', help='whether to choose negative from the same sequence')
    parser.add_argument('--smooth', action='store_true', help='whether to further smooth dynamics inputs')
    # Network params
    parser.add_argument('--e_dim', type=int, default=512, help='Dimension of encoders output')
    parser.add_argument('--conv_layers', type=int, default=0, help='Layer structure for ConvNeXt')
    parser.add_argument('--audio_style', type=int, choices=[1, 2, 3], help='1: one-shot style, 13 mfcc + 26 fbank, 2: 80 fbanks, 3: 26 ssc coefs')
    parser.add_argument('--e_x', type=str, choices=['resnet', 'conv1d_x', 'conv2d'], help='Landmarks encoder type')
    parser.add_argument('--e_a', type=str, choices=['resnet', 'conv1d_a', 'conv2d'], help='Audio encoder type')
    parser.add_argument('--sa_x', action='store_true', help='self attention after conv1?')
    parser.add_argument('--sa_a', action='store_true', help='self attention after conv1?')
    parser.add_argument('--sa_2d', action='store_true', help='self attention after conv1?')
    # Spatio-temporal resolution
    parser.add_argument('--smoothing_length', type=int, default=1)
    parser.add_argument('--pyramid_level', type=int, default=1)
    parser.add_argument('--pyramid_kernel_size', type=int, default=2)
    args = parser.parse_args()
    print('Arguments received, calling main !')
    main(args)