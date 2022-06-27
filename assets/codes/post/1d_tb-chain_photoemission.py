#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.style.use('dark_background')
mpl.rcParams['axes.unicode_minus'] = False

def photoem_mat_elem(k0, N=3):
    '''
    '''

    k0    = np.asarray(k0)
    nkpts = k0.size
    # photoemission matrix elements
    pk = np.array([
        np.sin(m*j*np.pi / (N+1))*np.exp(-1j*k0*j)
        for m in range(1, N+1)
        for j in range(1, N+1)
    ]).reshape((N, N, nkpts)).sum(axis=1)

    ek = np.array([
        -2*np.cos(m*np.pi/(N+1))
        for m in range(1, N+1)
    ])

    return ek, np.abs(pk)**2

if __name__ == "__main__":
    nkpts = 400      # No. of k-points
    nedos = 500      # No. of points for DOS
    sigma = 0.05     # energy broadening

    # k-space grid
    k0     = np.linspace(-np.pi, np.pi, nkpts)
    # energy grid
    e0     = np.linspace(-2-sigma*5, 2+sigma*5, nedos)
    x0, y0 = np.meshgrid(k0, e0, indexing='ij')

    ############################################################
    fig = plt.figure(
        figsize=(7.2, 4.8),
        dpi=480,
        constrained_layout=True
    )

    # 3x3 subplots
    axes_array = np.arange(9, dtype=int).reshape((3, 3))

    axes = fig.subplot_mosaic(
        axes_array,
        empty_sentinel=-1,
        gridspec_kw=dict(
            # height_ratios= [1, 0.5, 0.5],
            # width_ratios=[2, 2],
            # hspace=0.05,
            # wspace=0.06,
        )
    )
    axes = [axes[ii] for ii in range(axes_array.max()+1)]
    ############################################################

    # chain_lengths = np.arange(9) + 1
    # -1 for infinite chain length
    chain_lengths = np.array([1, 2, 3, 4, 5, 6, 10, 20, -1])

    for ii in range(chain_lengths.size):
        N  = chain_lengths[ii]
        ax = axes[ii]

        # finite chain length
        if N > 0:
            ek, pk = photoem_mat_elem(k0, N)
            # smearing
            ss = np.dot(
                pk.T,
                (1 / np.sqrt(2*np.pi) / sigma) * np.exp(-(e0[None, :] - ek[:, None])**2 / 2 / sigma**2)
            )
        # infinite chain length
        else:
            ek = -2 * np.cos(k0)
            # smearing
            ss = (1 / np.sqrt(2*np.pi) / sigma) * np.exp(-(e0[None, :] - ek[:, None])**2 / 2 / sigma**2)

        ax.pcolormesh(x0, y0, ss, cmap='magma')
        ax.plot(k0, -2*np.cos(k0), ls='--', lw=0.4)

        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))

        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([
            r'-$\,\frac{\pi}{a}$',
            r'-$\,\frac{\pi}{2a}$',
            '0',
            r'$\frac{\pi}{2a}$',
            r'$\frac{\pi}{a}$'
        ])

        if ii > 5:
            ax.set_xlabel(r'$k$')

        if ii % 3 == 0:
            ax.set_ylabel(r'Energy ($t$)', labelpad=5)

        if N > 0:
            ax.text(0.05, 0.05,
            # ax.text(0.50, 0.95,
                r'$N={}$'.format(N),
                # ha='center', va='top',
                ha='left', va='bottom',
                fontsize='small',
                transform=ax.transAxes
            )
        else:
            ax.text(0.05, 0.05,
            # ax.text(0.50, 0.95,
                r'$N=\infty$',
                # ha='center', va='top',
                ha='left', va='bottom',
                fontsize='small',
                transform=ax.transAxes
            )

    plt.savefig('arpes_1d_tb.png')
    from subprocess import call
    call('feh -xdF arpes_1d_tb.png'.split())
