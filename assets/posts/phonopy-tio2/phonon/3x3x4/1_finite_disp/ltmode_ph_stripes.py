#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

# plt.style.use('dark_background')
plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False

import os, yaml
try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

############################################################

def read_ph_yaml(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.xz' or ext == '.lzma':
        try:
            import lzma
        except ImportError:
            raise("Reading a lzma compressed file is not supported "
                  "by this python version.")
        with lzma.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    elif ext == '.gz':
        import gzip
        with gzip.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    else:
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=Loader)

    freqs   = []
    dists   = []
    qpoints = []
    labels  = []
    eigvec  = []
    Acell   = np.array(data['lattice'])
    Bcell   = np.array(data['reciprocal_lattice'])

    for j, v in enumerate(data['phonon']):
        if 'label' in v:
            labels.append(v['label'])
        else:
            labels.append(None)
        freqs.append([f['frequency'] for f in v['band']])
        if 'eigenvector' in v['band'][0]:
            eigvec.append([np.array(f['eigenvector']) for f in v['band']])
        qpoints.append(v['q-position'])
        dists.append(v['distance'])

    if all(x is None for x in labels):
        if 'labels' in data:
            ss = np.array(data['labels'])
            labels = list(ss[0])
            for ii, f in enumerate(ss[:-1,1] == ss[1:,0]):
                if not f:
                    labels[-1] += r'|' + ss[ii+1, 0]
                labels.append(ss[ii+1, 1])
        else:
            labels = []

    return (Bcell,
            np.array(dists),
            np.array(freqs),
            np.array(qpoints),
            data['segment_nqpoint'],
            labels, eigvec)


############################################################
# the phonon band
Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml('band.yaml')

assert E1, "PHONON EIGENVECTORs MUST NOT BE EMPTY!"
E1 = np.asarray(E1)
nqpts, nbnds, natoms, _, _ = E1.shape
# real and imaginary  part of the phonon polarization vector
preal, pimag = E1[..., 0], E1[..., 1]

RatioLT = np.zeros_like(F1)
for ii in range(nqpts):
    q = np.dot(Q1[ii], Bcell)
    # exclude Gamma point
    if np.linalg.norm(q) > 1E-10:
        RatioLT[ii,:] = np.linalg.norm(
            (preal[ii] + 1j * pimag[ii]) * q / np.linalg.norm(q),
            axis=(1, 2)
        )
    # exclude Gamma point
    else:
        RatioLT[ii,:] = 0.5


############################################################
fig = plt.figure(
    figsize=(7.2, 3.0),
    dpi=480,
    # constrained_layout=True,
)

ax = plt.subplot()

############################################################
norm = mpl.colors.Normalize(vmin=0, vmax=1)
s_m = mpl.cm.ScalarMappable(cmap='seismic', norm=norm)
s_m.set_array([RatioLT])

for ii in range(0, F1.shape[1]):
    ik = 0
    for nseg in B1:
        x = D1[ik:ik+nseg]
        y = F1[ik:ik+nseg, ii]
        z = RatioLT[ik:ik+nseg, ii]
        
        ax.plot(
            x, y,
            lw=1.2, color='k', alpha=0.6
        )

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments,
                            colors=[s_m.to_rgba(ww)
                                    for ww in (z[1:] + z[:-1])/2.]
                            )
        lc.set_linewidth(1.2)
        lc.set_alpha(0.6)
        ax.add_collection(lc)

        ik += nseg

for ii in np.cumsum(B1)[:-1]:
    ax.axvline(
        x=D1[ii], ls='--',
        color='gray', alpha=0.8, lw=0.5
    )

divider = make_axes_locatable(ax)
ax_cbar = divider.append_axes('right', size='3%', pad=0.02)
cbar = plt.colorbar(s_m, cax=ax_cbar, ticks=[0, 1], alpha=0.6)
cbar.set_ticklabels(['T', 'L'])

ax.set_xlim(D1.min(), D1.max())
ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
if L1:
    ax.set_xticklabels(L1)
ax.set_ylabel('Frequency (THz)', labelpad=5)


############################################################
plt.tight_layout(pad=0.5)
plt.savefig('lt_c.png')
from subprocess import call
call('feh -xdF lt_c.png'.split())
