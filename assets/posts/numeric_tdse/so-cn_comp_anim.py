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


def SplitOperatorKin(psi0, V, L, dt, N=100):
    '''
    Splitting technique for evolving the time-dependent Schrodinger equation.

    Kinetic referenced second-order Suzuki-Trotter expansion is used.
    '''

    from scipy.fft import fft, ifft, fftfreq

    psi0  = np.asarray(psi0)
    V     = np.asarray(V)
    ngrid = V.shape
    assert V.ndim == len(L)
    assert V.shape == psi0.shape

    # the kinetic operator
    ff = [fftfreq(n, 1/n) for n in V.shape]

    if V.ndim == 1:
        k2 = (2 * np.pi / L[0])**2 * ff[0]**2
    elif V.ndim == 2:
        kx, ky = np.meshgrid(ff[0], ff[1])
        k2 = (np.pi * 2 / L[0])**2 * kx**2 + \
             (np.pi * 2 / L[1])**2 * ky**2
    else:
        raise(NotImplementedError)

    # the kinetic operator by one time step
    Ut_full = np.exp(-1j * dt * k2 / 2.)

    # the potential operator by half time step
    Uv_half = np.exp(-1j * dt * V / 2.)

    # Store all the wavefunctions
    PSI_t = np.zeros(psi0.shape + (N,), dtype=complex)
    # the initial wavefunction
    PSI_t[..., 0] = psi0

    for n in range(N-1):
        PSI_t[:,n+1] = Uv_half *\
            ifft(
                Ut_full * fft(
                    Uv_half * PSI_t[..., n]
                )
            )

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
    k0     = 5         # the momentum of the wavepacket
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
    PSI0 = CrankNicolson(psi0, V, x, dt, N)
    PSI1 = SplitOperatorKin(psi0, V, [L], dt, N)

    ############################################################
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    mpl.rcParams['axes.unicode_minus'] = False
    plt.style.use('dark_background')
    ############################################################

    fig = plt.figure(
        figsize=(7.2, 3.0),
        dpi=100,
        # constrained_layout=True,
    )
    ax = plt.subplot()

    l0, = ax.plot(x, np.abs(PSI0[:,0]), lw=1.8,
            label='Crank-Nicolson', color='red')
    l1, = ax.plot(x, np.abs(PSI1[:,0]), lw=1.8, ls='-',
            label='Split-Operator', color='white')

    ax.legend(loc='upper right')
    
    time_stamp = ax.text(
        0.02, 0.95,
        r"$k_0={}\,$a.u.".format(k0) + "\n" + r'$t={:6.2f}\,$a.u.'.format(0),
        ha='left', va='top',
        family='monospace',
        transform=ax.transAxes,
    )

    ax.set_xlim(xmin, xmin+L)
    ax.set_xlabel(r'$x$ [Bohr]', labelpad=5)
    ax.set_ylabel(r'$|\psi(x)|$ [a.u.]', labelpad=5)

    plt.tight_layout()
    ############################################################

    def wfc_propagation(iframe):
        l0.set_ydata(np.abs(PSI0[:,iframe]))
        l1.set_ydata(np.abs(PSI1[:,iframe]))

        time_stamp.set_text(
            r"$k_0={}\,$a.u.".format(k0) + "\n" + r'$t={:6.2f}\,$a.u.'.format(dt * iframe),
        )

        return l0, l1, time_stamp,

    ani = animation.FuncAnimation(
        fig,
        wfc_propagation,
        interval=10,
        blit=True,
        repeat=True,
        frames=N,
    )
    ############################################################
    # with open("wfc_ani.html", 'w') as w:
    #     print(ani.to_jshtml(), file=w)

    # use "imagemagick" writer to make loop gif
    # ani.save('gaus_wfc_k{}.gif'.format(k0), writer='imagemagick')

    ani.save('comp_cn-so.gif'.format(k0))

    # plt.show()
