#!/usr/bin/env python

import numpy as np
################################################################################
nr    = 501
r0    = 0.75
omega = np.pi
t     = np.linspace(0, np.pi / omega, nr, endpoint=True)
dt    = t[1] - t[0]

rr    = -r0* np.cos(omega * t)
vv    =  r0 * omega * np.sin(omega * t)

alpha = 2.0
beta  = 0.20
gamma = 0.20
E1    = alpha * rr
E2    = -E1
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

E3 = -np.sqrt(alpha**2 * rr**2 + W0**2)
E4 = -E3

V3 = np.array([
    -W0 / (alpha*rr + np.sqrt(alpha**2 * rr**2 + W0**2)),
    np.ones(nr)
]).T
V3 /= np.linalg.norm(V3, axis=1)[:,None]
V4 = np.array([
    -W0 / (alpha*rr - np.sqrt(alpha**2 * rr**2 + W0**2)),
    np.ones(nr)
]).T
V4 /= np.linalg.norm(V4, axis=1)[:,None]

nac1 = vv * np.sum(
    V3 * np.matmul(Hderv, V4[...,None])[:,:,0], axis=1
) / (E4 - E3)

nac2 = np.sum((V3[:-1] * V4[1:]) - (V3[1:] * V4[:-1]), axis=1) / (2*dt)


################################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.unicode_minus'] = False

# plt.style.use('dark_background')
# plt.style.use('seaborn')

fig = plt.figure(
  figsize=(6.4, 3.2),
  dpi=200,
)
ax = plt.subplot()
################################################################################

ax.plot(rr, nac1, color='r',
        label=r'$\dfrac{\langle \psi_i |\nabla_R {\cal H}|\psi_j\rangle}{\varepsilon_j -\varepsilon_i}\cdot\dot{r}$')
ax.plot(rr[:-1], nac2, color='b', ls='--',
        label=r'$\dfrac{\langle \psi_i(t) |\psi_j(t+\Delta t)\rangle - \langle \psi_i(t+\Delta t) |\psi_j(t)\rangle}{2\Delta t}$')

ax.legend(
    loc='lower left', fontsize='small',
)
ax.grid(lw=0.5, ls='--', color='grey', alpha=0.5)
ax.set_xlim(-0.75, 0.75)

ax.set_xlabel('$R$ [arb. units]', labelpad=5)
ax.set_ylabel('NAC [arb. units]', labelpad=5)

################################################################################
plt.tight_layout(pad=1.0)
plt.savefig('nac_comp.png')
# plt.show()

from subprocess import call
call('feh -xdF nac_comp.png'.split())

