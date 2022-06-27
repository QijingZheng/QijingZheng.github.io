#!/usr/bin/env python

import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import splu
################################################################################

def gaussian_wavepacket(x, x0, k, sigma=0.1):
    '''
    One dimensional Gaussian wavepacket
    '''

    x = np.asarray(x)
    g = np.sqrt(1 / np.sqrt(np.pi) / sigma) * np.exp(-(x - x0)**2 / 2 / sigma**2)

    return np.exp(1j * k*(x-x0)) * g

def CrankNicolson(psi0, V, x, dt, N=100, print_norm=False):
    '''
    Crank-Nicolson method for the 1D Schrodinger equation.
    '''
    # No. of spatial grid points
    J  = x.size - 1
    dx = x[1] - x[0]

    # the external potential
    V = spa.diags(V)
    # the kinetic operator
    O = np.ones(J+1)
    T = (-1 / 2 / dx**2) * spa.spdiags([O, -2*O, O], [-1, 0, 1], J+1, J+1)

    # the two unitary matrices
    U2 = spa.eye(J+1) + (1j * 0.5 * dt) * (T + V)
    U1 = spa.eye(J+1) - (1j * 0.5 * dt) * (T + V)
    # splu requires CSC matrix format for efficient decomposition
    U2 = U2.tocsc()
    LU = splu(U2)

    # Store all the wavefunctions
    PSI_t = np.zeros((J+1, N), dtype=complex)
    # the initial wavefunction
    PSI_t[:, 0] = psi0

    for n in range(N-1):
        b            = U1.dot(PSI_t[:,n])
        PSI_t[:,n+1] = LU.solve(b)
        if print_norm:
            print(n, np.trapz(np.abs(PSI_t[:,n+1])**2, x))

    return PSI_t


################################################################################
if __name__ == "__main__":
    # length in Bohr
    L = 20
    # left boundary
    xmin = -L / 2.
    # No. of spatial grid points
    J = 999
    x = np.linspace(xmin, xmin+L, J+1, endpoint=True)
    dx = x[1] - x[0]

    # the gaussian wavepacket as initial wavefunction
    x0     = 0          # the center of the wavepacket
    k0     = 0          # the momentum of the wavepacket
    sigmax = 0.4        # the width of the wavepacket
    psi0   = gaussian_wavepacket(x, x0=x0, k=k0, sigma=sigmax)

    # time step in atomic units, 1 a.u. = 24.188 as
    dt = 2 / (2 * np.pi / sigmax + k0)**2
    print("Reasonable time step: {:.2E} a.u.".format(dt))
    # No. of temporal grid points
    N = int(2 / dt) + 1

    # the externial potentials
    V = np.zeros_like(x)

    # The time evolution of Schrodinger equation
    PSI = CrankNicolson(psi0, V, x, dt, N)

    ############################################################
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams['axes.unicode_minus'] = False
    plt.style.use('dark_background')
    ############################################################

    fig = plt.figure(
        figsize=(7.5, 3.0),
        dpi=300,
        # constrained_layout=True,
    )
    axes = [plt.subplot(1, 2, ii+1) for ii in range(2)]
    ############################################################
    ax = axes[0]
    for ii in range(N):
        ax.plot(
            x, np.abs(PSI[:,ii]),
            color=mpl.cm.bwr(ii / N),
            lw=0.4,
        )

    ax.set_xlim(xmin, xmin+L)
    ax.set_ylim(-0.05, 1.25)

    ax.set_xlabel(r'$x$ [Bohr]', labelpad=5)
    ax.set_ylabel(r'$|\psi(x)|$', labelpad=5)

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=0, vmax=N*dt),
            cmap=mpl.cm.bwr
        ),
        ax=ax,
        orientation='vertical',
        # label=r'$t$ [a.u.]',
        fraction=0.06, pad=0.02, extend='both', shrink=1.0,
        ticks=range(6),
        # location='top'
    )
    cbar.ax.set_title(r'$t$ [a.u.]', fontsize='x-small')

    ############################################################
    ax = axes[1]

    wfc = ax.pcolormesh(x, np.arange(N) * dt, np.abs(PSI).T, cmap='magma')

    ax.set_xlabel(r'$x$ [Bohr]', labelpad=5)
    ax.set_ylabel(r'$t$ [a.u.]', labelpad=5)

    cbar = fig.colorbar(
        wfc,
        ax=ax,
        orientation='vertical',
        # label=r'$|\psi(x)|$',
        fraction=0.06, pad=0.02, extend='both',
        # location='top'
    )
    cbar.ax.set_title(r'$|\psi(x)|$', fontsize='x-small')
    ############################################################
    plt.tight_layout()
    plt.savefig('wavepacket_momentum_{:d}.png'.format(k0))

    from subprocess import call
    call('feh -xdF wavepacket_momentum_{:d}.png'.format(k0).split())
