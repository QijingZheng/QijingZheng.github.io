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
    L = 40
    # left boundary
    xmin = -L / 2.
    # No. of spatial grid points
    J = 1500
    x = np.linspace(xmin, xmin+L, J+1, endpoint=True)
    dx = x[1] - x[0]

    # the gaussian wavepacket as initial wavefunction
    x0     = -7.5          # the center of the wavepacket
    sigmax = 0.4        # the width of the wavepacket

    # The time evolution of Schrodinger equation
    init_k0 = [2, 2]
    Vmax    = [10, 100]
    init_V0 = []
    for ii in range(2):
        V = np.zeros_like(x)
        barrier_width = 2.0
        V[(x >= 0) & (x <= barrier_width)] = Vmax[ii]
        init_V0 += [V]

    total_time = 3.0
    DTs = [2 / (2 * np.pi / sigmax + k0)**2 for k0 in init_k0]
    PSI =[
        CrankNicolson(
            gaussian_wavepacket(x, x0=x0, k=k0, sigma=sigmax),
            V0,
            x, dt,
            int(total_time / dt) + 1
        )
        for dt, k0, V0 in zip(DTs, init_k0, init_V0)
    ]
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
    axes = [plt.subplot(1, 2, ii+1) for ii in range(2)]

    l0, = axes[0].plot(x, np.abs(PSI[0][:,0]), lw=1.0, color=mpl.cm.bwr(0))
    l1, = axes[1].plot(x, np.abs(PSI[1][:,0]), lw=1.0, color=mpl.cm.bwr(0))

    txts = [
        axes[ii].text(
            0.02, 0.95,
            r"$k_0={}\,$a.u.".format(init_k0[ii]) + "\n" 
            r"$V_0={}\,$a.u.".format(Vmax[ii]) + "\n" + r'$t={:6.2f}\,$a.u.'.format(0),
            ha='left', va='top',
            family='monospace',
            transform=axes[ii].transAxes,
        )
        for ii in range(2)
    ]
    txt0, txt1 = txts

    for ii in range(2):
        ax = axes[ii]
        ax.set_xlim(xmin, xmin+L)
        ax.set_xlabel(r'$x$ [Bohr]', labelpad=5)
        if ii == 0:
            ax.set_ylabel(r'$|\psi(x)|$ [a.u.]', labelpad=5)

        ax_t = ax.twinx()
        ax_t.set_ylim(-0.5, 15.5)

        ax_t.fill_between(x, init_V0[ii], color='w', lw=0)
        if ii == 1:
            ax_t.set_ylabel(r'$V$ [a.u.]', labelpad=5)
            ax_t.yaxis.tick_right()

    plt.tight_layout()
    ############################################################
    # time step in animations
    dt_a = 0.01

    def wfc_propagation(iframe):
        lines = [l0, l1]
        for jj in range(2):
            l = lines[jj]
            kk = int(dt_a * iframe / DTs[jj])
            l.set_ydata(np.abs(PSI[jj][:,kk]))
            l.set_color(mpl.cm.bwr(iframe * dt_a / total_time))

        for jj in range(2):
            txt = txts[jj]
            txt.set_text(
                r"$k_0={}\,$a.u.".format(init_k0[jj]) + "\n" 
                r"$V_0={}\,$a.u.".format(Vmax[jj]) + "\n" +
                r'$t={:6.2f}\,$a.u.'.format(dt_a * iframe),
            )

        return *lines, *txts

    ani = animation.FuncAnimation(
        fig,
        wfc_propagation,
        interval=10,
        blit=True,
        repeat=True,
        frames=int(total_time / dt_a),
    )
    ############################################################
    # with open("wfc_ani.html", 'w') as w:
    #     print(ani.to_jshtml(), file=w)

    # use "imagemagick" writer to make loop gif
    # ani.save('gaus_wfc_k{}.gif'.format(k0), writer='imagemagick')

    ani.save('gaus_wfc_bar_2x1.gif')

    # plt.show()
