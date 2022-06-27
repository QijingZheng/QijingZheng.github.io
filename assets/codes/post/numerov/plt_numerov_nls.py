#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint, simps

def radial_wfc_numerov(r0, n=1, l=0, Z=1, du=0.001):
    '''
    Numerov algorithm
    
                  [12 - 10f(n)]*y(n) - y(n-1)*f(n-1)
        y(n+1) = ------------------------------------
                               f(n+1)

    where
        
        f(n) = 1 + (h**2 / 12)*g(n)

        g(n) = [E + (2*Z / x) - l*(l+1) / x**2]

    here, we use reverse integration from the other end.
    '''
    # the wavefunction
    ur = np.zeros(r0.size)
    fn = np.zeros(r0.size)

    E      = -float(Z)**2 / n**2
    ur[-1] = 0.0
    ur[-2] = du

    dr  = r0[1] - r0[0]
    h12 = dr**2 / 12.

    gn = (E + 2*Z / r0 - l*(l+1) / r0**2)
    fn = 1. + h12 * gn

    for ii in range(r0.size - 3, -1, -1):
        ur[ii] = (12 - 10*fn[ii+1]) * ur[ii+1] - \
                 ur[ii+2] * fn[ii+2]
        ur[ii] /= fn[ii]

    # normalization
    ur /= np.sqrt(simps(ur**2, x=r0))

    return ur


if __name__ == "__main__":
    ################################################################################
    r0  = np.linspace(1E-10, 80, 1000)

    nls = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2)
    ]

    from sympy import lambdify
    from sympy.abc import r, Z
    from sympy.physics.hydrogen import R_nl

    ur0 = [
        lambdify((r, Z), r * R_nl(n, l, r, Z), 'numpy')(r0, 1) for n, l in nls
    ]
    deltas = [0.01, -0.01, 0.01, 0.01, -0.01, 0.01]
    ur2 = [radial_wfc_numerov(r0, n, l, du=deltas[ii]) for ii, (n, l) in enumerate(nls)]

    ################################################################################
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')
    mpl.rcParams['axes.unicode_minus'] = False
    ################################################################################
    fig = plt.figure(
        figsize=(9.0, 3.6),
        dpi=300
    )
    axes = [plt.subplot(2, 3 ,ii+1) for ii in range(len(nls))]

    #################################################################################

    for ii in range(len(nls)):
        ax = axes[ii]

        ax.plot(
            r0, ur2[ii],
            ls='none',
           ms=3, marker='o', mfc='white', mew=1.0, 
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'][ii],
            zorder=1,
            label=r'Numerov',
        )

        ax.plot(r0, ur0[ii], lw=0.6, color='cyan', zorder=2,
                label=r'Exact')

        ax.set_xlabel(r'$r$ [$a_0$]',   labelpad=5)
        ax.set_ylabel(r'$u_{{ {}{} }}(r)$'.format(*nls[ii]), labelpad=5)

        ax.legend(loc='best', fontsize='small', ncol=1)

        ax.set_xlim(-2, 16 + ii * 8)

    plt.tight_layout()
    plt.savefig('fig2.png')
    plt.show()
