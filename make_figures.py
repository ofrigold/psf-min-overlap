import argparse
import json
import os
import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PIL import Image
from optics import bfp_grids, z_planes, zstack_psfs, crop

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--out', default=None)
parser.add_argument('--gif', action='store_true')
args = parser.parse_args()

with open(os.path.join(args.run, 'params.json')) as f:
    p = json.load(f)
out = args.out or os.path.join(args.run, 'figures')
os.makedirs(out, exist_ok=True)
g = bfp_grids(p)
z = z_planes(p)
circ = g['circ'].numpy() > 0
rows, cols = np.where(circ)
r0, r1, c0, c1 = rows.min() - 4, rows.max() + 5, cols.min() - 4, cols.max() + 5
z_rel = (z - p['NFP']).numpy()
loss = np.load(os.path.join(args.run, 'loss.npy'))
blur = torchvision.transforms.GaussianBlur(p['blur_kernel'], sigma=p['blur_sigma'])


def psfs(mask):
    with torch.no_grad():
        return crop(blur(zstack_psfs(mask, z, g, p['nphotons']).unsqueeze(1)).squeeze(1), p['H'], p['H']).numpy()


def phase(mask):
    return np.where(circ, np.angle(np.exp(1j * mask.numpy())), np.nan)[r0:r1, c0:c1]


def scalebar(ax, um=2):
    ax.add_artist(AnchoredSizeBar(ax.transData, um / p['pixel_size_FOV'], '%d $\\mu m$' % um, 'lower right',
                                  pad=0.4, color='white', frameon=False, size_vertical=0.6))


def show_psf(fig, ax, img, title=None):
    im = ax.imshow(img)
    fig.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title)


def show_mask(ax, mask, cbar=True):
    im = ax.imshow(phase(mask), vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted', interpolation='nearest')
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[-np.pi, 0, np.pi])
        cb.ax.set_yticklabels(['$-\\pi$', '0', '$\\pi$'])


flat = torch.zeros(p['Hmask'], p['Hmask'])
best = torch.load(os.path.join(args.run, 'mask_best.pt'))
n = p['n_planes']

fig, axes = plt.subplots(2, n + 2, figsize=(2.9 * (n + 2), 5.6), subplot_kw={'xticks': [], 'yticks': []})
for r, (mask, label) in enumerate([(flat, 'No mask'), (best, 'Learned mask')]):
    ps = psfs(mask)
    show_mask(axes[r, 0], mask, cbar=False)
    axes[r, 0].set_ylabel(label, fontsize=13)
    for j in range(n):
        show_psf(fig, axes[r, j + 1], ps[j], 'z = %.1f $\\mu m$' % z_rel[j] if r == 0 else None)
    show_psf(fig, axes[r, n + 1], ps.sum(0), 'Sum of %d planes' % n if r == 0 else None)
    scalebar(axes[r, n + 1])
axes[0, 0].set_title('Phase mask')
plt.tight_layout()
plt.savefig(os.path.join(out, 'concept.png'), dpi=100)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), subplot_kw={'xticks': [], 'yticks': []})
show_mask(axes[0], best)
axes[0].set_title('Learned phase mask')
show_psf(fig, axes[1], psfs(best).sum(0), 'Sum of %d PSFs' % n)
scalebar(axes[1])
plt.tight_layout()
plt.savefig(os.path.join(out, 'mask_and_sum.png'), dpi=100)
plt.close()

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(loss, lw=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Overlap loss (mean Jaccard)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out, 'loss.png'), dpi=100)
plt.close()

if args.gif:
    last = int(np.argmin(loss)) - 1
    epochs = sorted(set(list(range(0, min(30, last))) + list(range(30, last, 4)) + [last]))
    steps = [(None, 0)] + [(e, e + 1) for e in epochs]
    frames = []
    for e, it in steps:
        mask = flat if e is None else torch.load(os.path.join(args.run, 'masks_per_epoch', 'mask_epoch_%d.pt' % e))
        fig = plt.figure(figsize=(13.5, 4.4), dpi=82)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.28, left=0.03, right=0.97, top=0.86,
                              bottom=0.14)
        ax0 = fig.add_subplot(gs[0], xticks=[], yticks=[])
        show_mask(ax0, mask)
        ax0.set_title('Phase mask')
        ax1 = fig.add_subplot(gs[1], xticks=[], yticks=[])
        show_psf(fig, ax1, psfs(mask).sum(0), 'Sum of %d PSFs' % n)
        scalebar(ax1)
        ax2 = fig.add_subplot(gs[2])
        ax2.semilogy(np.arange(len(loss)), loss, color='0.8', lw=1.5)
        ax2.semilogy(np.arange(it + 1), loss[:it + 1], color='C0', lw=2)
        ax2.plot(it, loss[it], 'o', color='C3', ms=7)
        ax2.set_xlim(0, len(loss))
        ax2.set_ylim(loss.min() * 0.6, loss.max() * 1.6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Overlap loss (mean Jaccard)')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        fig.suptitle('Iteration %d' % it, fontsize=15)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    pal = frames[-1].quantize(colors=255, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    frames[0].save(os.path.join(out, 'convergence.gif'), save_all=True, append_images=frames[1:],
                   duration=[90] * (len(frames) - 1) + [2500], loop=0, optimize=True)
