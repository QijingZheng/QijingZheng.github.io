#!/usr/bin/env python

import numpy as np


rmax  = 0.75
nr    = 501
rr    = np.linspace(-rmax, rmax, nr, endpoint=True)
alpha = 2.0
beta  = 0.20
gamma = 0.20
E1    = alpha * rr
E2    = -alpha * rr
W0    = gamma * np.exp(-rr**2 / beta**2)

# The Hamiltonian
Ham = np.zeros((nr, 2, 2))
Ham[:,0,0] = E1
Ham[:,1,1] = E2
Ham[:,0,1] = W0
Ham[:,1,0] = W0

# The derivative of Hamiltonian to R
Hderv = np.zeros((nr, 2, 2))
Hderv[:,0,0] = alpha
Hderv[:,1,1] = -alpha
Hderv[:,0,1] = -2 * rr * W0 / beta**2
Hderv[:,1,0] = -2 * rr * W0 / beta**2

vals, vecs = np.linalg.eigh(Ham)

oo = np.random.randint(2, size=(nr))
xx = oo - 1

E3 = vals[range(nr),oo]
E4 = vals[range(nr),xx]

V3 = vecs[range(nr),:,oo]
V4 = vecs[range(nr),:,xx]

nac = np.sum(V3 * np.matmul(Hderv, V4[...,None])[:,:,0], axis=1) / (E4 - E3)

rho_1 = V3[:,0]**2
rho_2 = V4[:,0]**2

################################################################################
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection 

# plt.style.use('dark_background')
# plt.style.use('seaborn')

fig = plt.figure(
  figsize=plt.figaspect(0.50),
  # constrained_layout=True,
  dpi=200,
)
# plt.subplots_adjust(wspace=0.3)
# axes = [plt.subplot(2, 1, ii+1) for ii in range(2)]
axes = [
    plt.subplot(2,2,1),
    plt.subplot(2,2,2),
    plt.subplot(2,1,2),
]
################################################################################
ax = axes[0]

ax.plot(rr, E1, lw=0.3, ls='--', color='r', label=r'$\varepsilon_1^d=\alpha\cdot R$')
ax.plot(rr, E2, lw=0.3, ls='--', color='b', label=r'$\varepsilon_2^d=-\alpha\cdot R$')

ax.plot(rr, E3, lw=0.3, ls='-', color='r')
ax.plot(rr, E4, lw=0.3, ls='-', color='b')

ax.set_ylabel('Energy [arb. units]', labelpad=5)

ax = axes[1]
ax.plot(rr, rho_1, color='r', lw=0.3, label=r'$|c_{11}|^2$')
ax.plot(rr, rho_2, color='b', lw=0.3, label=r'$|c_{21}|^2$')

ax.set_ylabel('$|c_{i1}|^2$ [arb. units]', labelpad=5)

ax = axes[2]
ax.plot(rr, 5*W0, lw=0.5, color='g', ls='--',
        label=r'$5\times\lambda$')
ax.plot(rr, 0.2*nac, lw=0.5, color='m',
        label=r'$0.2\times d_{12}$')

ax.set_ylabel('Energy [arb. units]', labelpad=5)


leg_loc = ['lower center', 'center right', 'upper right']
for ii, ax in enumerate(axes):
    ax.grid(lw=0.5, ls='--', color='grey', alpha=0.5)
    ax.set_xlim(-0.75, 0.75)

    ax.set_xlabel('$R$ [arb. units]', labelpad=5)

    ax.legend(loc=leg_loc[ii],)

################################################################################
plt.tight_layout(pad=1.0)
plt.savefig('num_nac_noorder_1.png')
# plt.show()

from subprocess import call
call('feh -xdF num_nac_noorder_1.png'.split())

