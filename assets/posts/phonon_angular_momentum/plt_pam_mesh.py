#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ggplot')
mpl.rcParams['axes.unicode_minus'] = False

pam_mesh = np.loadtxt('pam-gridsize.dat')

fig = plt.figure(
  figsize=(6.4, 3.6),
  dpi=300,
)
ax = plt.subplot()

ax.axhline(y=0, ls='--', color='gray', lw=0.5)

for ii in range(3):
    ax.plot(
        pam_mesh[:,0], pam_mesh[:,ii+1],
        marker='o', mew=1.2, mfc='w', ms=6,
        ls='-.', # lw=0.5,
        label='$J_{}$'.format('xyz'[ii])
    )

ax.legend(loc='upper right', numpoints=2)

ax.set_xlabel('Mesh Size', labelpad=5)
ax.set_ylabel(r'PAM [$\hbar$]', labelpad=5)

plt.tight_layout()
plt.savefig('total_pam_size.png')
plt.show()

