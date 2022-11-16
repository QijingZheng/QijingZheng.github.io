#!/usr/bin/env python

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# plt.style.use('dark_background')
plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False

import ase
from ase.io import read

# from phonopy.units import THzToCm
THzToCm = 33.3564095198152

############################################################
Natoms = len(read('POSCAR'))
mode = 1
# Normal mode coordinate
Qmode   = np.loadtxt('m_{:02d}/Q.dat'.format(mode))

if not os.path.isfile('Epes_{:02d}.dat'.format(mode)):
    Emode = np.array([
        read('m_{:02d}/{:02d}/OUTCAR'.format(mode, ii)).get_potential_energy()
        for ii in range(1, 16)
    ])
    Emode -= Emode.min()
    np.savetxt('Epes_{:02d}.dat'.format(mode), Emode, fmt='%12.6f')
else:
    Emode = np.loadtxt('Epes_{:02d}.dat'.format(mode))

# frequencies in cm-1 from OUTCAR
Wmode   = 783.495252

Qharm = np.linspace(Qmode.min(), Qmode.max(), 100)
# Qharm = [np.linspace(-1, 1, 100)
Eharm = 0.5 * (Qharm * 1E-10 * np.sqrt(ase.units._mp))**2 * \
        (Wmode / THzToCm * 1E12 * 2 * np.pi)**2 /\
        ase.units._e * 1000

############################################################
fig = plt.figure(
    figsize=(3.6, 2.4),
    dpi=480,
    constrained_layout=True
)

ax = plt.subplot()

############################################################
ax.plot(Qmode, Emode*1000/Natoms, ls='none',
        marker='o', ms=4, mew=1.0, mfc='w',
        color='b',
        label=r'$\omega={:.2f}\,cm^{{-1}}$'.format(Wmode))
ax.plot(Qharm, Eharm/Natoms, ls='--', lw=0.7, color='r', label=r'$Q\,\omega^2/\,2$')

# ax.grid('on', ls='-', lw=0.2, alpha=0.3)

ax.legend(loc='upper center', fontsize='x-small')

ax.set_xlabel(r'Q [$\sqrt{\mathrm{amu}}\cdot\AA$]',
              labelpad=5, fontsize='small')

ax.set_ylabel('Energy per atom [meV]', labelpad=5, fontsize='small')
############################################################


plt.savefig('pes.png')
from subprocess import call
call('feh -xdF pes.png'.split())
