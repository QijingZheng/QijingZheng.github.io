#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
plt.style.use('ggplot')

from scipy.special import ellipe

def ellipe_expansion(m):
    return 1 + m * (0.463 - 0.25 * np.log(m))

m0 = np.linspace(0.00001, 0.1, 20)
f1 = ellipe(1 - m0)
f2 = ellipe_expansion(m0)

figure = plt.figure(
    figsize=(4, 2.5),
    dpi=300,
)
ax = plt.subplot()

ax.plot(m0, f1, ls='-', lw=1, color='r', alpha=0.6, label='scipy.special.ellipe')
ax.plot(m0, f2, ls='none',
        marker='o', ms=4, mfc='w', mew=1.2,
        color='blue', #alpha=0.6,
        label='expansion')

ax.set_xlabel(r'$\lambda$', labelpad=5)
ax.grid('on', ls='--', lw=0.4, alpha=0.5)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('ellipe_exp.png')
plt.show()
