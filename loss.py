import torch
import torchvision
from optics import crop


def footprints(psfs, p):
    x = torchvision.transforms.GaussianBlur(p['blur_kernel'], sigma=p['blur_sigma'])(psfs.unsqueeze(1)).squeeze(1)
    x = crop(x, p['H'], p['H'])
    mx, mn = x.amax(dim=(-2, -1), keepdim=True), x.amin(dim=(-2, -1), keepdim=True)
    return torch.sigmoid(x - (mx - mn) / 2)


def pairwise_jaccard(b, eps=1e-6):
    f = b.flatten(1)
    inter = f @ f.T
    s = f.sum(1)
    return inter / (s[:, None] + s[None, :] - inter + eps)


def overlap_loss(psfs, p):
    j = pairwise_jaccard(footprints(psfs, p))
    i, k = torch.triu_indices(len(j), len(j), offset=1)
    return j[i, k].mean()
