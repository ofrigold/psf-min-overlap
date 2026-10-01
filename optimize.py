import argparse
import json
import os
import time
from datetime import datetime
import numpy as np
import torch
from params import get_params
from optics import bfp_grids, z_planes, zstack_psfs
from loss import overlap_loss

parser = argparse.ArgumentParser()
parser.add_argument('--preset', default='lattice_680')
parser.add_argument('--out', default=None)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--max_epochs', type=int, default=None)
parser.add_argument('--save_every_epoch', action='store_true')
args = parser.parse_args()

p = get_params(args.preset)
if args.max_epochs is not None:
    p['max_epochs'] = args.max_epochs
out = args.out or os.path.join('results', '%s_%s' % (args.preset, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
os.makedirs(os.path.join(out, 'masks_per_epoch') if args.save_every_epoch else out, exist_ok=True)
with open(os.path.join(out, 'params.json'), 'w') as f:
    json.dump({**p, 'preset': args.preset}, f, indent=2)

torch.manual_seed(p['seed'])
dev = torch.device(args.device)
g = bfp_grids(p, dev)
z = z_planes(p, dev)
mask = torch.nn.Parameter(torch.zeros(p['Hmask'], p['Hmask'], device=dev))
torch.save(mask.data.cpu(), os.path.join(out, 'mask_init.pt'))
opt = torch.optim.Adam([mask], lr=p['lr'])

losses, best, stall, t0 = [], float('inf'), 0, time.time()
for epoch in range(p['max_epochs']):
    opt.zero_grad()
    loss = overlap_loss(zstack_psfs(mask, z, g, p['nphotons']), p)
    loss.backward()
    losses.append(loss.item())
    if losses[-1] < best - p['min_delta']:
        best, stall = losses[-1], 0
        torch.save(mask.data.cpu(), os.path.join(out, 'mask_best.pt'))
    else:
        stall += 1
    opt.step()
    if args.save_every_epoch:
        torch.save(mask.data.cpu(), os.path.join(out, 'masks_per_epoch', 'mask_epoch_%d.pt' % epoch))
    print('epoch %d | loss %.5f | best %.5f | %.1fs' % (epoch, losses[-1], best, time.time() - t0), flush=True)
    if stall >= p['patience']:
        break

np.save(os.path.join(out, 'loss.npy'), np.array(losses))
with torch.no_grad():
    best_mask = torch.load(os.path.join(out, 'mask_best.pt')).to(dev)
    np.save(os.path.join(out, 'psfs_best.npy'), zstack_psfs(best_mask, z, g, p['nphotons']).cpu().numpy())
print('done:', out)
