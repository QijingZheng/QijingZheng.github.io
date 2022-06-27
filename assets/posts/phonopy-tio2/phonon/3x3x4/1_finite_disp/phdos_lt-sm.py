#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.style.use('dark_background')
# plt.style.use('ggplot')
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
    figsize=(7.2, 3.6),
    dpi=480,
    constrained_layout=True
)

layout = np.arange(2, dtype=int).reshape((-1, 1))
axes   = fig.subplot_mosaic(
    layout,
    empty_sentinel=-1,
    gridspec_kw=dict(
        # height_ratios= [1.0],
        # width_ratios=[1, 1.0],
        # hspace=0.05,
        # wspace=0.06,
    )
)
axes = np.array([ax for ax in axes.values()])

############################################################

# the phonon dos
Q2, W2, F2 = read_ph_yaml('mesh_nac.yaml', mode='mesh')
Fs, Dos_sm = ph_dos_smearing(F2, W2, sigma=0.02)
Fl, Dos_lt = np.loadtxt('total_dos.dat').T
############################################################
for ii in range(2):
    ax = axes[ii]
    ax.grid('on', which='both', ls=':', lw=0.3, color='gray', zorder=0)
    ax.set_xlim(0, 25)

    ax.set_ylabel('DOS (arb. unit)', labelpad=5)
    if ii == 1:
        ax.set_xlabel('Frequency (THz)', labelpad=5)

ax = axes[0]
ax.fill_between(
    Fs, Dos_sm, lw=0, fc='r',
    alpha=1.0, zorder=1,
    label=r'Gaussian Smearing: $\sigma=0.02$'
)
ax.legend()


ax = axes[1]
ax.fill_between(
    Fl, Dos_lt, lw=0, fc='y',
    alpha=1.0, zorder=1,
    label=r'Linear Tetrahedron: $\Delta E = 0.02$'
)
ax.legend()

############################################################


plt.savefig('dos_comp.png')
from subprocess import call
call('feh -xdF dos_comp.png'.split())
