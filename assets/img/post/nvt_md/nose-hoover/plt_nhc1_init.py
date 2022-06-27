#!/usr/bin/env python

from subprocess import call
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False


T0      = 0.1
xv_init = np.mgrid[0:3, 0:3].reshape((2, -1)).T * np.sqrt(2*T0)
xp_dist = np.load('nhc-1_init.npy').reshape((9, -1, 5))

plt.style.use('ggplot')

figure = plt.figure(
    figsize=(12, 12),
    dpi=300
)
axes = [plt.subplot(3,3,ii+1) for ii in range(xv_init.shape[0])]

for l in range(9):
    ax      = axes[l]
    x0, v0  = xv_init[l]
    md_data = xp_dist[l]
    
    print("Averaged Temperature: {:12.6E}".format(2*np.average(md_data[:,2])))


    ax.scatter(
        md_data[:, 0], md_data[:, 1],
        # c=np.exp(-np.linalg.norm(md_data[:, :2], axis=1)**2),
        c='g',
        facecolor='w',
        marker='.', s=0.5,  # mew=0.8, mfc='w',
        cmap='Greens',
        # alpha=0.5
    )

    # ax.hist2d(
    #     md_data[:,0], md_data[:,1], bins=(100, 100),
    #     range=[[-1.2, 1.2], [-1.2, 1.2]],
    #     # norm=mpl.colors.LogNorm()
    # )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.set_xticks(np.arange(-1.0, 1.1, 0.5))
    ax.set_yticks(np.arange(-1.0, 1.1, 0.5))

    ax.set_xlabel('$x$ [arb. unit]', fontsize='large')
    ax.set_ylabel('$p$ [arb. unit]', fontsize='large')

    ax.text(0.50, 0.98,
            r'$x_0 = {:.2f};\quad v_0 = {:.2f}$'.format(x0, v0),
            color='k',
            va='top', ha='center',
            # fontsize='small',
            transform=ax.transAxes)


    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    xax = divider.append_axes("top", size="33%", pad=0.15, sharex=ax)
    yax = divider.append_axes("right", size="33%", pad=0.15, sharey=ax)

    # make some labels invisible
    # xax.xaxis.set_tick_params(labelbottom=False)
    # yax.yaxis.set_tick_params(labelleft=False)

    xax.axis('off')
    yax.axis('off')

    x0 = np.linspace(-1.5, 1.5, 1000)
    y0 = (1 / 2 / np.pi / 0.1)**(1./2) * np.exp(-x0**2 / (2*0.1))

    xax.hist(md_data[:, 0], bins=100,
            histtype='stepfilled', ec='k',
            color='b', alpha=0.3, density=True)
    xax.plot(x0, y0, color='b', lw=2.0, alpha=0.7)

    yax.hist(md_data[:, 1], bins=100,
            histtype='stepfilled', ec='k',
            color='r', alpha=0.3,
            orientation='horizontal', density=True)
    yax.plot(y0, x0, color='r', lw=2.0, alpha=0.7)

    # xax.text(0.95, 0.05,
    #          # r'$\rho(x) = \left({\frac{m\omega^2}{2\pi kT}}\right)^{\frac{1}{2}} \,e^{\mathrm{-}{\frac{m\omega^2 x^2}{2kT}}}$',
    #          r'$\rho(x) = \sqrt{{\frac{m\omega^2}{2\pi kT}}} \,e^{\mathrm{-}{\frac{m\omega^2 x^2}{2kT}}}$',
    #          color='b',
    #          va='bottom', ha='left',
    #          # fontsize='small',
    #          transform=xax.transAxes)
    #
    # yax.text(0.05, 1.00,
    #          # r'$\rho(p) = \left({\frac{1}{2\pi mkT}}\right)^{\frac{1}{2}} \,e^{\mathrm{-}{\frac{p^2}{2mkT}}}$',
    #          r'$\rho(p) = \sqrt{{\frac{1}{2\pi mkT}}} \,e^{\mathrm{-}{\frac{p^2}{2mkT}}}$',
    #          color='r',
    #          # rotation=-90,
    #          va='top', ha='left',
    #          # fontsize='small',
    #          transform=yax.transAxes)

plt.tight_layout(pad=1)
plt.savefig('xp_nhc1_init.png')
# plt.show()

call('feh -xdF xp_nhc1_init.png'.split())
