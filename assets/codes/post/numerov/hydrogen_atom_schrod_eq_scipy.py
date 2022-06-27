#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint, simps

def radial_schrod_deriv(u, r, l, E, Z=1):
    '''
    The radial Schrodinger equation for hydrogen-like atom is

        u''(x) - [(l(l+1)/x^2) - (2Z/x) - E]u(x) = 0 

    Since it is a second-order equation, i.e. 
    
        y'' + g(x)y(x) = 0

    To use scipy.integrate.odeint, we can turn this equation into two
    first-order equation by defining a new dependent variables
    
        y'(x) = z(x)
        z'(x) = -g(x)y(x)

    Then we can solve this system of ODEs using "odeint" wit list. 
    '''
    
    y, z = u

    return np.array([
        z, 
        ((l*(l+1) / r**2) - (2*Z / r) - E) * y
    ])


def radial_wfc_scipy(r0, n=1, l=0, Z=1, direction='F', du=0.1):
    '''
    Get the radial wavefunction by integrating the equation with
    scipy.integrate.odeint.
    '''

    assert direction.upper() in ['F', 'B']

    E = -float(Z) / n**2

    # forward integration 
    if direction.upper() == 'F':
        ur = odeint(radial_schrod_deriv, [0.0, du], r0, args=(l, E, Z))[:,0]

    # back integration 
    else:
        ur = odeint(radial_schrod_deriv, [0.0, -du], r0[::-1], args=(l, E, Z))[:,0][::-1]

    ur /= np.sqrt(simps(ur**2, x=r0))

    return ur

if __name__ == "__main__":
    ################################################################################
    # Define the radial grid
    r0 = np.linspace(1E-10, 20, 500)

    ################################################################################
    # The energy levels of hydrogen like atoms 
    #
    #    E = - Z / n**2
    #
    # where "n" is the main quantum number and energy is in unit of Rydberg.
    ################################################################################
    Z = 1
    l = 0
    n = 1

    ur1 = radial_wfc_scipy(r0, n, l, direction='F')
    ur2 = radial_wfc_scipy(r0, n, l, direction='B')

    ################################################################################
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
    fig = plt.figure(
        figsize=(4.8, 2.4),
        dpi=300
    )
    ax  = plt.subplot()

    ax.plot(
        r0, ur1,
        ls='none',
        ms=4, marker='o', mfc='white', mew=1.0, 
        zorder=1,
        label=r'Forward integration',
    )
    
    ax.plot(
        r0, ur2,
        ls='none',
        ms=4, marker='o', mfc='white', mew=1.0, 
        zorder=1,
        label=r'Backward integration',
    )
    ax.plot(r0, 2*r0*np.exp(-r0), lw=1.0, color='k', zorder=2, label=r'Exact: $2r\cdot e^{-r}$')
    
    ax.set_xlabel(r'$r$ [$a_0$]',   labelpad=5)
    ax.set_ylabel(r'$u(r)$ [a.u.]', labelpad=5)

    ylim = list(ax.get_ylim())
    ylim[1] = 1.0
    ax.set_ylim(ylim)

    ax.legend(loc='best', fontsize='x-small')

    plt.tight_layout()
    plt.savefig('v1.png')
    plt.show()
    ################################################################################
