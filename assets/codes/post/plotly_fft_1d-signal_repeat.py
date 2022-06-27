#!/usr/bin/env python

import numpy as np
from scipy.fftpack import fft, fftfreq

# No. of data points
N = 800
# The length of the signal
L = 100
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

############################################################
# plot the results using Plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=2, cols=2,
    horizontal_spacing = 0.15
)
fig.update_layout(
    width=800, height=600
)
############################################################
fig.add_trace(
    go.Scatter(x=x0, y=y0, mode='lines+markers', showlegend=False),
    row=1, col=1)
fig.add_trace(
    go.Scatter(x=x1, y=y1, mode='lines+markers', showlegend=False),
    row=2, col=1)
fig.add_trace(
    go.Scatter(x=w0, y=f0, mode='markers', showlegend=False),
    row=1, col=2)
fig.add_trace(
    go.Scatter(x=w1, y=f1, mode='markers', showlegend=False),
    row=2, col=2)

fig.update_xaxes(title_text="Time [s]", range=[0, 2*L], row=1, col=1)
fig.update_xaxes(title_text="Time [s]", range=[0, 2*L], row=2, col=1)
fig.update_xaxes(title_text="Frequence [Hz]", range=[0, 0.25], row=1, col=2)
fig.update_xaxes(title_text="Frequence [Hz]", range=[0, 0.25], row=2, col=2)

fig.update_yaxes(title_text=r'$g(t)$', title_font=dict(size=20), row=1, col=1)
fig.update_yaxes(title_text=r'$g(t)$', title_font=dict(size=20), row=2, col=1)
fig.update_yaxes(title_text=r'$\mathcal{F}\{g\}$', title_font=dict(size=20), row=1, col=2)
fig.update_yaxes(title_text=r'$\mathcal{F}\{g\}$', title_font=dict(size=20), row=2, col=2)

fig.update_traces(
    marker=dict(
        size=4, color='white',
        line=dict(width=0.5, color='red')
    ),
    line=dict(color='red', width=0.5),
    row=1, col=1
)
fig.update_traces(
    marker=dict(
        size=4, color='white',
        line=dict(width=0.5, color='blue')
    ),
    line=dict(color='blue', width=0.5),
    row=2, col=1
)
fig.update_traces(
    marker=dict(
        size=6, color='white',
        line=dict(width=0.8, color='red')
    ),
    row=1, col=2
)
fig.update_traces(
    marker=dict(
        size=6, color='white',
        line=dict(width=0.8, color='blue')
    ),
    row=2, col=2
)
############################################################
fig.write_html('plotly_fft.html', include_plotlyjs=False, full_html=False)
fig.show()
