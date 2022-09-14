#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

plt.style.use('dark_background')
# plt.style.use('ggplot')

nr    = 501
rr    = np.linspace(-1, 1, nr, endpoint=True)
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
E5 = vals[range(nr),xx]
E6 = vals[range(nr),oo]

# band reorder --- wavefunction
V5 = vecs[range(nr),:,xx]
V6 = vecs[range(nr),:,oo]

# phase correction
V5[V5[:,1] < 0] *= -1
V6[V6[:,1] < 0] *= -1

nac1 = np.sum(V3 * np.matmul(Hderv, V4[...,None])[:,:,0], axis=1) / (E4 - E3)
nac2 = np.sum(V5 * np.matmul(Hderv, V6[...,None])[:,:,0], axis=1) / (E6 - E5)


fig = plt.figure(
  figsize=plt.figaspect(0.60),
  dpi=200,
)
ax = plt.subplot()

ax.plot(rr, E1, lw=2.0, ls='-', color='r', label=r'$\varepsilon_1^d=\alpha\cdot R$')
ax.plot(rr, E2, lw=2.0, ls='-', color='b', label=r'$\varepsilon_2^d=-\alpha\cdot R$')
ax.plot(rr, E5, lw=2.5, ls=':', color='r', label=r'$\varepsilon_1^a$')
ax.plot(rr, E6, lw=2.5, ls=':', color='b', label=r'$\varepsilon_2^a$')

ax.plot(rr, 5*W0, lw=1.0, color='g',
        label=r'$\lambda = \gamma\cdot e^{-r^2/\beta^2}$')
ax.plot(rr, 0.4*nac1, lw=1.0, color='m',
        # label=r'$\dfrac{\langle \phi_i |\nabla_R {\cal H}|\phi_j\rangle}{\varepsilon_j -\varepsilon_i}$ (no-reorder)')
        label=r'NAC w/o REORDER')
ax.plot(rr, 0.4*nac2, lw=1.0, color='c',
        # label=r'$\dfrac{\langle \phi_i |\nabla_R {\cal H}|\phi_j\rangle}{\varepsilon_j -\varepsilon_i}$ (reorder)')
        label=r'NAC w/ REORDER')

ax.legend(
    loc='upper right', # fontsize='small',
    # ncol=3, # labelspacing=3,
)

ax.grid(lw=0.5, ls='--', color='grey', alpha=0.5)
ax.set_xlim(-0.75, 0.75)

ax.set_xlabel('$R$ [arb. units]', labelpad=5)
ax.set_ylabel('Energy [arb. units]', labelpad=5)

plt.tight_layout()
plt.savefig('ham.png')
# plt.show()

from subprocess import call
call('feh -xdF ham.png'.split())

