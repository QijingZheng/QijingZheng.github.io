#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.style.use('dark_background')
# plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False

import ase
from ase.io import read

# from phonopy.units import THzToCm
THzToCm = 33.3564095198152

############################################################
ph_modes = np.array([16, 17, 18], dtype=int)
rundir   = ["m_{}".format(ii) for ii in ph_modes]
# Normal mode coordinate
Qmodes   = np.array([np.loadtxt('{}/Q.dat'.format(d)) for d in rundir])

# # PES along the normal mode
# Emodes   = []
# for m in ph_modes:
#     if not os.path.isfile('Epes_{}.dat'.format(m)):
#         tmp = np.array([read('{:02d}/OUTCAR').get_potential_energy() for ii in range(1, 16)])
#         np.savetxt('Epes_{}.dat'.format(m), tmp, fmt='%12.6f')
#     else:
#         tmp = np.loadtxt('Epes_{}.dat'.format(m))
#     Emodes.append(tmp)
# Emodes = np.asarray(Emodes)

# frequencies in cm-1 from OUTCAR
Wmodes   = np.array([35.658442, 35.658442, 87.694598])

Qharm = np.array([np.linspace(x.min(), x.max(), 100) for x in Qmodes])
# Qharm = np.array([np.linspace(-1, 1) for x in Qmodes])
Eharm = 0.5 * (Qharm * 1E-10 * np.sqrt(ase.units._mp))**2 * \
        (Wmodes[:, None] / THzToCm * 1E12 * 2 * np.pi)**2 /\
        ase.units._e * 1000

marker = ['o', 's', 'H']
colors = ['r', 'g', 'b']
############################################################
fig = plt.figure(
    figsize=(7.2, 2.4),
    dpi=480,
    constrained_layout=True
)

layout = np.arange(len(ph_modes), dtype=int).reshape((1, -1))
axes   = fig.subplot_mosaic(
    layout,
    empty_sentinel=-1,
    gridspec_kw=dict(
        # height_ratios= [1.0],
        # width_ratios=[1, 0.4],
        # hspace=0.05,
        # wspace=0.06,
    )
)
axes = np.array([ax for ax in axes.values()])

############################################################
for ii in range(len(ph_modes)):
    ax = axes[ii]
    Q0 = Qharm[ii]
    E0 = Eharm[ii]

    ax.plot(Q0, -E0, ls='--', lw=0.7, color=colors[ii])

    # ax.set_xlim(-2.1, 2.1)
    ax.grid('on', ls='-', lw=0.2, alpha=0.3)

for ax in axes:
    ax.set_xlabel(r'Q [$\sqrt{\mathrm{amu}}\cdot\AA$]',
                  labelpad=5, fontsize='small')

axes[0].set_ylabel('Energy [meV]', labelpad=5, fontsize='small')
############################################################
############################################################


plt.savefig('pes.png')
from subprocess import call
call('feh -xdF pes.png'.split())
