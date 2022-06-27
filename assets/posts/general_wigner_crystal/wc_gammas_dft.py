#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.style.use('ggplot')
# plt.style.use('dark_background')
################################################################################
Ndat    = 500
Lm      = np.linspace(5, 40, Ndat, endpoint=True)
nu      = np.linspace(0.05, 4, Ndat, endpoint=True)
M       = np.array([4.26, 2.78, 9.61, 6.81])
g       = 2.0
a0      = 0.529
gam_s0  = 29.5

d1        = 4.80   # CoCl2
d2        = 3.40   # HOPG
epsilon_1 = 1.558  # CoCl2
epsilon   = (d1 + d2) * epsilon_1 / d1

nu_wc_max = M[:,None] * (g * Lm[None,:] / a0 / epsilon / gam_s0)**2 * (np.sqrt(3) / 2 / np.pi)
nu_mott   = np.sqrt(3) / 2 * (Lm[None,:] * M[:,None] / a0 / epsilon)**2

################################################################################
fig = plt.figure(
    figsize=(4.5, 3.6),
    dpi=300,
    # constrained_layout=True
)
ax = plt.subplot()
################################################################################

for ii in range(len(M)):
    ax.plot(Lm, nu_wc_max[ii],
            color=mpl.cm.viridis((M[ii] - M.min()) / M.max()),
            ls='-', lw=1.0,
            label=r'$\gamma_s=29.5;\enspace m^*\!={}$'.format(M[ii]))

    # ax.plot(Lm, nu_mott[ii],
    #         color=mpl.cm.viridis((M[ii] - M.min()) / M.max()),
    #         ls='--', lw=1.0,
    #         label=r'Mott Criterion ($m^\ast\! = {}$)'.format(M[ii]))

ax.axvspan(xmin=6, xmax=12, color='blue', lw=0, alpha=0.1)
print(ax.get_legend_handles_labels())
################################################################################
ax.legend(
    loc='upper left',
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
plt.savefig('gamma_s_dft.png')
from subprocess import call
call('feh -xdF gamma_s_dft.png'.split())

# plt.show()
################################################################################
