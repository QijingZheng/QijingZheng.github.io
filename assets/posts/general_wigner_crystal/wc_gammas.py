#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# plt.style.use('ggplot')
# plt.style.use('dark_background')
################################################################################
Ndat    = 500
Lm      = np.linspace(5, 100, Ndat, endpoint=True)
nu      = np.linspace(0.05, 4, Ndat, endpoint=True)
m       = 0.5
g       = 2.0
a0      = 0.529
gam_s0  = 29.5

d1        = 4.80   # CoCl2
d2        = 3.40   # HOPG
epsilon_1 = 4      # CoCl2
epsilon   = (d1 + d2) * epsilon_1 / d1

nu_wc_max = (g * m * Lm / a0 / epsilon / gam_s0)**2 * (np.sqrt(3) / 2 / np.pi)
nu_wc_max1 = (g * 1 * Lm / a0 / epsilon / gam_s0)**2 * (np.sqrt(3) / 2 / np.pi)
nu_wc_max2 = (g * 2 * Lm / a0 / epsilon / gam_s0)**2 * (np.sqrt(3) / 2 / np.pi)
nu_mott   = np.sqrt(3) / 2 * (Lm * m / a0 / epsilon)**2

x, y  = np.meshgrid(Lm, nu)
Gam_s = (g * m * x / (a0 * epsilon * np.sqrt(2*np.pi*y/np.sqrt(3))))
################################################################################
fig = plt.figure(
    figsize=(4.0, 3.6),
    dpi=300,
    # constrained_layout=True
)
ax = plt.subplot()
################################################################################
gmap = ax.pcolor(x, y, Gam_s, cmap='viridis')
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
cbar.ax.set_title(
    r'$\gamma_s$',
    # fontsize='small'
)

ax.plot(Lm, nu_wc_max,
        color='b', ls='--', lw=1.0,
        label=r'$\gamma_s=29.5;\enspace m^*\!=0.5$')
ax.plot(Lm, nu_wc_max1,
        color='w', ls='--', lw=1.0,
        label=r'$\gamma_s=29.5;\enspace m^*\!=1.0$')
ax.plot(Lm, nu_wc_max2,
        color='m', ls='--', lw=1.0,
        label=r'$\gamma_s=29.5;\enspace m^*\!=2.0$')
ax.plot(Lm, nu_mott,
        color='r', ls='--', lw=1.0,
        label=r'Mott Criterion ($m^\ast\! = 0.5$)')

################################################################################
ax.legend(
    loc='upper right',
    fontsize='x-small'
)

ax.set_xlim(Lm.min(), Lm.max())
ax.set_ylim(nu.min(), nu.max())

ax.set_xlabel(r'$\lambda_m$ [$\AA$]', labelpad=5)
ax.set_ylabel(r'Filling Factor $(\nu)$', labelpad=5)

ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))

################################################################################
plt.tight_layout(pad=0.5)
plt.savefig('gamma_s.png')
from subprocess import call
call('feh -xdF gamma_s.png'.split())

# plt.show()
################################################################################
