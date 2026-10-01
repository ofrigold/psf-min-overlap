PRESETS = {
    'lattice_680': dict(lamda=0.68, NA=0.75, M=40, f_4f=100e3, n_imm=1., n_sample=1., pixel_size_CCD=11,
                    pixel_size_SLM=24, Hmask=387, H=51, n_planes=7, dz=4.7, nphotons=4000),
    'atom_array': dict(lamda=0.852, NA=0.55, M=32, f_4f=58e4, n_imm=1., n_sample=1., pixel_size_CCD=11,
                       pixel_size_SLM=20, Hmask=1283, H=91, n_planes=7, dz=4.7, nphotons=2500),
}
COMMON = dict(lr=0.05, max_epochs=2000, patience=10, min_delta=1e-5, blur_kernel=9, blur_sigma=2., seed=999)


def get_params(name):
    p = {**COMMON, **PRESETS[name]}
    p['NFP'] = (p['n_planes'] - 1) * p['dz'] / 2
    p['pixel_size_FOV'] = p['pixel_size_CCD'] / p['M']
    return p
