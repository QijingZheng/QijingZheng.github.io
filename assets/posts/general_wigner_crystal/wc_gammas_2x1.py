#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# plt.style.use('ggplot')
plt.style.use('dark_background')
################################################################################
Ndat    = 500
Lm      = np.linspace(6, 100, Ndat, endpoint=True)
nu      = np.linspace(0.1, 4, Ndat, endpoint=True)
m       = 0.28
g       = 2.0
a0      = 0.529
epsilon = 2.9
gam_s0  = 29.5

nu_wc_max = (g * m * Lm / a0 / epsilon / gam_s0)**2 * (np.sqrt(3) / 2 / np.pi)
nu_mott   = np.sqrt(3) / 2 * (Lm * m / a0 / epsilon)**2

x, y  = np.meshgrid(Lm, nu)
Gam_s = (g * m * x / (a0 * epsilon * np.sqrt(2*np.pi*y/np.sqrt(3))))
ne    = y / (np.sqrt(3) / 2 * x**2)
################################################################################
fig = plt.figure(
    figsize=(7.2, 3.0),
    dpi=300,
    # constrained_layout=True
)
# ax = plt.subplot()
axes = [plt.subplot(1, 2, ii+1) for ii in range(2)]
################################################################################
ax = axes[0]
gmap = ax.pcolor(x, y, Gam_s, cmap='magma')
cbar = fig.colorbar(
    gmap,
    ax=ax,
    orientation='vertical',
    fraction=0.06, pad=0.02,
    # extend='both',
    shrink=1.0,
    # ticks=range(6),
    # location='top'
)
cbar.ax.set_title(r'$\gamma_s$', fontsize='small')

ax = axes[1]
gmap = ax.pcolor(x, y, ne, cmap='viridis')
cbar = fig.colorbar(
    gmap,
    ax=ax,
    orientation='vertical',
    fraction=0.06, pad=0.02,
    # extend='both',
    shrink=1.0,
    # ticks=range(6),
    # location='top'
)
# cbar.ax.set_title(r'$n_e$ [$10^{-', fontsize='small')

for ax in axes:
    ax.plot(Lm, nu_wc_max, color='w', ls='--', lw=1.0, label=r'$\gamma_s = 29.5$')
    ax.plot(Lm, nu_mott, color='r', ls='--', lw=1.0, label='Mott Criterion')

################################################################################
for ax in axes:
    ax.legend(
        loc='upper right',
        fontsize='small'
    )

    ax.set_xlim(Lm.min(), Lm.max())
    ax.set_ylim(nu.min(), nu.max())

    ax.set_xlabel(r'$\lambda_m$ [$\AA$]', labelpad=5)
    ax.set_ylabel(r'Filling Factor $(\nu)$', labelpad=5)

    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))

################################################################################
plt.tight_layout(pad=0.5)
plt.savefig('kaka.png')
from subprocess import call
call('feh -xdF kaka.png'.split())

# plt.show()
################################################################################
