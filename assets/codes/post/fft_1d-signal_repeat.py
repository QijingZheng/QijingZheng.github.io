#!/usr/bin/env python

import numpy as np
from scipy.fftpack import fft, fftfreq

# No. of data points
N   = 800
# The length of the signal
L   = 100
# the dampling parameter of the signal
tau = 10

x0 = np.linspace(0, L, N, endpoint=False)
y0 = np.cos((x0 - L/2.)) * np.exp(-(x0-L/2)**2 / tau**2)

# repeat the signal 
x1 = np.r_[x0, x0 + L]
y1 = np.r_[y0, y0]

# The frequency
dx = x0[1] - x0[0]
w0 = fftfreq(N, dx)
w1 = fftfreq(N*2, dx)

# Perform FFT power spectrum density and normalize the results
f0 = np.abs(fft(y0) / L)**2
f1 = np.abs(fft(y1) / (2*L))**2

# plot the results

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig, axes = plt.subplots(
    nrows=2, ncols=2,
    dpi=300, figsize=plt.figaspect(0.75)
)

############################################################
# plot the signal
ax = axes[0, 0]
ax.plot(x0, y0, ls='-', lw=0.2, color='r',
        marker='o', ms=1.5, mew=0.3, mfc='w')
ax.set_xlim(0, 2*L)

ax = axes[1, 0]
ax.plot(x1, y1, ls='-', lw=0.2, color='b',
        marker='*', ms=1.5, mew=0.3, mfc='w')

for ax in axes[:,0]:
    ax.set_xticks([0, L, 2*L])
    ax.set_xlabel('Time [s]', labelpad=5)
    ax.set_ylabel('$g(t)$', labelpad=5)

############################################################
# plot the PSD
ax = axes[0, 1]
ax.plot(w0, f0, ls='-', lw=0, color='r',
        marker='o', ms=2, mew=0.5, mfc='w')
ax.vlines(w0, ymin=0, ymax=f0, ls='-', lw=0.5, color='r')

ax = axes[1, 1]
ax.plot(w1, f1, ls='--', lw=0, color='b',
        marker='*', ms=3, mew=0.5, mfc='w')
ax.vlines(w1, ymin=0, ymax=f1, ls='-', lw=0.5, color='b')

for ax in axes[:,1]:
    ax.set_xlim(0.00, 0.25)
    ax.set_xlabel('Frequency [Hz]', labelpad=5)
    ax.set_ylabel(r'$\mathcal{F}\{g\}$', labelpad=5)

# save and show the figure
plt.tight_layout(pad=1)
plt.savefig('fft.png')
plt.show()
