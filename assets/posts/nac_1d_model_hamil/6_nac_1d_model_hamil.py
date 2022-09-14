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

vals, vecs = np.linalg.eig(Ham)
E3, E4 = vals.T
V3 = vecs[...,0]
V4 = vecs[...,1]

oo = np.asarray(E3 <= E4, dtype=int)
xx = oo - 1
# band reorder --- energy
E3 = vals[range(nr),xx]
E4 = vals[range(nr),oo]

# band reorder --- wavefunction
V3 = vecs[range(nr),:,xx]
V4 = vecs[range(nr),:,oo]

# phase correction
V3[V3[:,1] < 0] *= -1
V4[V4[:,1] < 0] *= -1

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
  figsize=plt.figaspect(1.00),
  # constrained_layout=True,
  dpi=200,
)
# plt.subplots_adjust(wspace=0.3)
axes = [plt.subplot(2, 1, ii+1) for ii in range(2)]
################################################################################
ax = axes[0]

ax.plot(rr, E1, lw=1.0, ls='--', color='r', label=r'$\varepsilon_1^d=\alpha\cdot R$')
ax.plot(rr, E2, lw=1.0, ls='--', color='b', label=r'$\varepsilon_2^d=-\alpha\cdot R$')

sm = mpl.cm.ScalarMappable(cmap='seismic', norm=mpl.colors.Normalize(0,1))
for y, z in zip([E3, E4], [rho_1, rho_2]):
    sm.set_array(z)

    pp = np.array([rr, y]).T.reshape((-1, 1, 2))
    ss = np.concatenate([pp[:-1], pp[1:]], axis=1)
    lc = LineCollection(ss, colors=[sm.to_rgba(xx) for xx in (z[1:] + z[:-1])/2])
    lc.set_linewidth(3)
    ax.add_collection(lc)

# ax.plot(rr, E3, lw=2.5, ls=':', color='r', label=r'$\varepsilon_1^a$')
# ax.plot(rr, E4, lw=2.5, ls=':', color='b', label=r'$\varepsilon_2^a$')

ax.plot(rr, 5*W0, lw=1.0, color='g', ls='--',
        # label=r'$\lambda = \gamma\cdot e^{-r^2/\beta^2}$')
        label=r'$5\times\lambda$')
ax.plot(rr, 0.2*nac, lw=1.0, color='m',
        # label=r'$\dfrac{\langle \phi_i |\nabla_R {\cal H}|\phi_j\rangle}{\varepsilon_j -\varepsilon_i}$')
        label=r'$0.2\times d_{12}$')

ax.legend(
    loc='lower center', fontsize='small',
    ncol=4, # labelspacing=3,
    mode='expand',
    handlelength=1.5,
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.05), 
)

ax.grid(lw=0.5, ls='--', color='grey', alpha=0.5)
ax.set_xlim(-0.75, 0.75)

ax.set_xlabel('$R$ [arb. units]', labelpad=5)
ax.set_ylabel('Energy [arb. units]', labelpad=5)
############################################################
ax = axes[1]

# ax.plot(np.abs(nac), W0)
# ax.plot(rr, np.abs(nac * (E4 - E3)),
#         label=r'${\langle \phi_i |\nabla_R {\cal H}|\phi_j\rangle}$')
# ax.plot(rr, np.abs(E4 - E3),
#         label=r'${|\varepsilon_j -\varepsilon_i|}$')

ax.plot(rr, rho_1, color='r',
        label=r'$|c_{11}|^2$')
ax.plot(rr, rho_2, color='b',
        label=r'$|c_{21}|^2$')

ax.legend(loc='center right', fontsize='small')
ax.grid(lw=0.5, ls='--', color='grey', alpha=0.5)

ax.set_xlabel('$R$ [arb. units]', labelpad=5)
ax.set_ylabel('$|c_{i1}|^2$ [arb. units]', labelpad=5)

################################################################################
plt.tight_layout(pad=1.0)
plt.savefig('num_nac_order_pha.png')
# plt.show()

from subprocess import call
call('feh -xdF num_nac_order_pha.png'.split())

