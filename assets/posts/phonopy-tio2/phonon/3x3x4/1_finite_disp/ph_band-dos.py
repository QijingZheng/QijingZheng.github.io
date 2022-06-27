#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# plt.style.use('dark_background')
plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False

import os, yaml
try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

############################################################

def read_ph_yaml(filename, mode='band'):
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

    if mode == 'band':
        freqs   = []
        dists   = []
        qpoints = []
        labels  = []
        for j, v in enumerate(data['phonon']):
            if 'label' in v:
                labels.append(v['label'])
            else:
                labels.append(None)
            freqs.append([f['frequency'] for f in v['band']])
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


        return (np.array(dists),
                np.array(freqs),
                np.array(qpoints),
                data['segment_nqpoint'],
                labels)

    elif mode == 'mesh':
        qpoints = []
        qpt_wht = []
        freqs   = []
        for j, v in enumerate(data['phonon']):
            freqs.append([f['frequency'] for f in v['band']])
            qpoints.append(v['q-position'])
            qpt_wht.append(v['weight'])

        # normalize the q-points weights
        qpt_wht = np.array(qpt_wht) / np.sum(qpt_wht)

        return np.array(qpoints), np.array(qpt_wht), np.array(freqs)
    else:
        pass

def ph_dos_smearing(freqs, whts, sigma=0.5, nedos=500):
    '''
    Gaussian smearing of the DOS
    '''

    nqpts, nbnds = freqs.shape
    assert nqpts == whts.shape[0]
    
    fmin = freqs.min()
    fmax = freqs.max()
    f0   = np.linspace(
        fmin - 10 * sigma,
        fmax + 10 * sigma,
        nedos, endpoint=True
    )

    nfac = 1. / np.sqrt(np.pi * 2) / sigma

    return f0, nfac * np.sum(
            whts * np.sum(
                np.exp(-(f0[:,None,None] - freqs[None,:,:])**2 / sigma**2), 
                axis=2
            ), axis=1
        )


############################################################
fig = plt.figure(
    figsize=(6.4, 3.6),
    dpi=480,
    constrained_layout=True
)

layout = np.arange(2, dtype=int).reshape((1, -1))
axes   = fig.subplot_mosaic(
    layout,
    empty_sentinel=-1,
    gridspec_kw=dict(
        height_ratios= [1.0],
        width_ratios=[1, 0.4],
        # hspace=0.05,
        # wspace=0.06,
    )
)
axes = np.array([ax for ax in axes.values()])

############################################################
# the phonon band
D1, F1, Q1, B1, L1 = read_ph_yaml('band.yaml')

# the phonon dos
# Q2, W2, F2         = read_ph_yaml('mesh.yaml', mode='mesh')
# Fsmear, Dos        = ph_dos_smearing(F2, W2, sigma=0.2)
Fsmear, Dos = np.loadtxt('total_dos.dat').T
############################################################
ax = axes[0]

for ii in range(0, F1.shape[1]):
    ik = 0
    for nseg in B1:
        ax.plot(
            D1[ik:ik+nseg],
            F1[ik:ik+nseg,ii],
            lw=0.5, color='r', alpha=0.8
        )
        ik += nseg

for ii in np.cumsum(B1)[:-1]:
    ax.axvline(
        x=D1[ii], ls='--',
        color='gray', alpha=0.8, lw=0.5
    )

ax.set_xlim(D1.min(), D1.max())
# ax.set_ylim(-6, 26)
ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
if L1:
    ax.set_xticklabels(L1)
fmin, fmax = ax.get_ylim()

ax = axes[1]
# ax.plot(
#     Dos, Fsmear,
#     lw=0.1, color='r'
# )
ax.fill_betweenx(
    Fsmear, Dos, lw=0, fc='b',
    alpha=0.6,
)

ax.set_ylim(fmin, fmax)

############################################################
for ax in axes:
    ax.set_ylabel('Frequency (THz)', labelpad=5)

axes[1].set_xlabel('DOS (arb. unit)', labelpad=5)
axes[1].set_xticklabels([])

axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

############################################################


plt.savefig('ph.png')
from subprocess import call
call('feh -xdF ph.png'.split())
