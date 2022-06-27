#!/usr/bin/env python

import numba
import numpy as np
from fMandel import fmandelbrot

############################################################
# Mandelbrot set from numpy
############################################################


def mandel(ext, nx=1000, ny=1000, nmax=1000):
    '''
    '''

    xmin, xmax, ymin, ymax = ext
    x, y = np.mgrid[xmin:xmax:1j*nx, ymin:ymax:1j*nx]
    z0 = x + 1j * y
    z = np.zeros_like(z0)

    d = nmax * np.ones(z0.shape, dtype=np.int32)
    m = d > 0

    for ii in range(nmax):
        z[m] = z[m]**2 + z0[m]

        m, m1 = np.abs(z) < 2, m
        d[m ^ m1] = ii

    return x, y, d

############################################################
# Speedup by using numba
############################################################


@numba.njit(cache=True, fastmath=True)
def mandel_numba(xmin, xmax, ymin, ymax, nmax=1000, nx=1000, ny=1000):
    dat = np.ones((nx, ny)) * nmax

    for ii in range(nx):
        for jj in range(ny):
            x0 = xmin + (xmax - xmin) * ii / (nx - 1.)
            y0 = ymin + (ymax - ymin) * jj / (ny - 1.)
            z0 = x0 + 1j*y0

            z = 0j
            for kk in range(nmax):
                if np.abs(z) > 2.:
                    dat[ii, jj] = kk
                    break
                z = z**2 + z0

    return dat


def ax_update(ax):
    '''
    '''
    # set to False, otherwise infinite loop
    ax.set_autoscale_on(False)

    # xmin, ymin, xran, yran = ax.viewLim.bounds
    # xmax = xmin + xran
    # ymax = ymin + yran

    v = ax.viewLim
    xmin, xmax, ymin, ymax = v.x0, v.x1, v.y0, v.y1

    # get new data according to the selected extent
    t0 = time.time()
    # m = mandel_numba(xmin, xmax, ymin, ymax)

    m = fmandelbrot([xmin, xmax, ymin, ymax], 1000, 1000)
    t1 = time.time()
    print("Elapsed Time: {:8.4f} [sec]".format(t1 - t0))

    # set the data
    img = ax.images[-1]
    img.set_data(m.T)
    img.set_extent([xmin, xmax, ymin, ymax])

    # finally redraw
    ax.figure.canvas.draw_idle()


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    figure = plt.figure()
    ax = plt.subplot()

    t0 = time.time()
    # x, y, m = mandel([-2,1, -1, 1])
    # m = mandel_numba(-2, 1, -1, 1)

    m = fmandelbrot([-2, 1, -1, 1], 1000, 1000)

    t1 = time.time()

    print("Elapsed Time: {:8.4f} [sec]".format(t1 - t0))

    ax.imshow(m.T, extent=(-2, 1, -1, 1), origin='lower', aspect=1.0)

    ax.callbacks.connect('xlim_changed', ax_update)
    ax.callbacks.connect('ylim_changed', ax_update)

    plt.show()
