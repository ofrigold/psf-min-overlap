import numpy as np
import torch
import torch.nn.functional as F


def bfp_grids(p, device='cpu'):
    N = p['Hmask']
    px_per_um = 1 / p['pixel_size_SLM']
    bfp_diam_px = 2 * p['f_4f'] * p['NA'] / np.sqrt(p['M'] ** 2 - p['NA'] ** 2) * px_per_um
    ccd_size_px = p['lamda'] * p['f_4f'] * px_per_um / p['pixel_size_CCD']
    pad = int(np.round(0.5 * (ccd_size_px - N)))

    x = np.linspace(-1, 1, N) * p['NA'] / p['n_imm'] * N / bfp_diam_px
    xx, yy = np.meshgrid(x, x)
    sin_imm = np.sqrt(xx ** 2 + yy ** 2)
    circ = (sin_imm < p['NA'] / p['n_imm']).astype('float32')
    cos_imm = np.sqrt(1 - (sin_imm * circ) ** 2) * circ
    defocus = np.exp(1j * 2 * np.pi * p['n_imm'] / p['lamda'] * (-p['NFP']) * cos_imm)

    sin_s = p['n_imm'] / p['n_sample'] * sin_imm
    circ_s = (sin_s < 1).astype('float32')
    cos_s = np.sqrt(1 - (sin_s * circ_s) ** 2) * circ_s
    kz = 2 * np.pi * p['n_sample'] / p['lamda'] * cos_s

    t = lambda a: torch.from_numpy(a).to(device)
    return {'circ': t(circ), 'kz': t(kz), 'defocus': t(defocus), 'pad': pad}


def z_planes(p, device='cpu'):
    return torch.arange(p['n_planes'], device=device, dtype=torch.float64) * p['dz']


def zstack_psfs(mask, z, g, nphotons):
    pad = g['pad']
    pupil = torch.exp(1j * z[:, None, None] * g['kz']) * g['defocus'] * g['circ']
    field = F.pad(pupil, (pad, pad, pad, pad)) * torch.exp(1j * F.pad(mask, (pad, pad, pad, pad)))
    img = torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm='forward'),
                                       dim=(-2, -1))) ** 2
    img = img * (nphotons / img.sum(dim=(-2, -1)))[:, None, None]
    return F.pad(img, (-pad, -pad, -pad, -pad)).float()


def crop(x, h, w):
    H, W = x.shape[-2:]
    y0, x0 = H // 2 - h // 2, W // 2 - w // 2
    return x[..., y0:y0 + h, x0:x0 + w]
