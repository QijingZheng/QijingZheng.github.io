#!/usr/bin/env python

from subprocess import call
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

################################################################################
# MD data: x, p, Ek, Ev, Etot
md_data = np.loadtxt('nhc-2_xp.dat')
# No. of time steps
nsw     = md_data.shape[0]
# time step
dt      = 0.1
# Time [ t / (2*pi / w) ]
t0      = np.arange(nsw) * dt / np.pi / 2
# Instantaneous temperature 2 * Ek
Tk      = md_data[:,2] * 2
# Cumulative averaged Temperature
T0      = np.cumsum(Tk) / np.arange(1, nsw+1)

################################################################################
plt.style.use('ggplot')

figure = plt.figure(
    figsize=(4.8, 2.4),
    dpi=300
)
ax = plt.subplot()

ax.plot(t0, Tk, 'k', lw=0.5)
ax.plot(t0, T0, 'r', lw=0.5)

ax.axhline(y=0.1, lw=0.5, color='b', ls='--')

ax.set_xlim(-0.5, 50.5)
ax.set_ylim(-0.02, 1.0)

ax.set_xlabel(r'$\omega t \,/\, 2\pi$')
ax.set_ylabel('Temperature')

plt.tight_layout(pad=1)
plt.savefig('temp_fluc_nhc2.png')
# plt.show()

call('feh -xdF temp_fluc_nhc2.png'.split())
