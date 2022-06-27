#!/usr/bin/env python

from subprocess import call
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')
################################################################################
if os.path.isfile('nhc2_qmass.dat'):
    xp_dist = np.loadtxt('nhc2_qmass.dat').reshape((3, -1, 5))[:,-10000:,:]
else:
    xp_dist = np.load('nhc2_qmass.npy').reshape((3, -1, 5))[:,2000:,:]

N = xp_dist.shape[1]

################################################################################

figure = plt.figure(
    figsize=(7.5, 2.5),
    dpi=300
)
axes = [plt.subplot(1, 3, ii+1) for ii in range(3)]

dt = 0.1
t0 = np.arange(N) * dt / np.pi / 2

gamma = [
    r'$Q_1 = \frac{Nk_BT}{(\omega_0 / 4)^2}$',
    r'$Q_1 = \frac{Nk_BT}{\omega_0^2}$',
    r'$Q_1 = \frac{Nk_BT}{(4\omega_0)^2}$'
]

for ii in range(3):
    ax = axes[ii]
    xx = xp_dist[ii]

    l1, = ax.plot(t0, xx[:, -1],
                  lw=0.0, color='k')
    l2, = ax.plot(t0, xx[:, -2],
                  lw=0.1, color='b')

    ax.fill_between(t0, y1=0, y2=xx[:, -2],
                    color=l2.get_color(),
                    alpha=0.3)
    ax.fill_between(t0, y1=xx[:, -2], y2=xx[:, -1],
                    color='r',
                    alpha=0.3)

    ax.set_xlabel(r'$\omega t \,/\, 2\pi$')
    if ii == 0:
        ax.set_ylabel('Energy [arb. unit]')

    ax.text(0.05, 0.98,
            # r'$x_0 = {:.2f};\quad v_0 = {:.2f}$'.format(x0, v0),
            gamma[ii],
            color='k',
            va='top', ha='left',
            # fontsize='small',
            transform=ax.transAxes)

    if ii > 0:
        ax.set_yticklabels([])

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.01, 0.4)

    if ii == 0:
        legend_elements = [
            Patch(facecolor='r', edgecolor='r', lw=0, label='$K$', alpha=0.3),
            Patch(facecolor='b', edgecolor='b', lw=0, label='$V$', alpha=0.3)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')


plt.tight_layout(pad=1)
plt.savefig('kv_qmass.png')
# plt.show()

call('feh -xdF kv_qmass.png'.split())
