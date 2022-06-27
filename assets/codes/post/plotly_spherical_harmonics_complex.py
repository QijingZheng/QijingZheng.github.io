#!/usr/bin/env python

import numpy as np
from sph_harm import sph_r, sph_c

################################################################################
N    = 50
u    = np.linspace(0, np.pi, N)
v    = np.linspace(0, 2*np.pi, N)
u, v = np.meshgrid(u, v)

x0   = np.sin(u) * np.cos(v)
y0   = np.sin(u) * np.sin(v)
z0   = np.cos(u)
xyz  = np.c_[x0.ravel(), y0.ravel(), z0.ravel()]

# Real spherical harmonics
LMAX = 3
SH   = [sph_c(xyz, l=l).reshape((N,N,-1)) for l in range(LMAX+1)]

################################################################################
import plotly.graph_objects as go
from plotly.subplots import make_subplots

nrows = LMAX + 1
ncols = 2*LMAX + 1

fig = make_subplots(
    rows=nrows, cols=ncols,
    specs=[
        [{'type': 'surface'} for jj in range(ncols)]
        for ii in range(nrows)
    ],
    subplot_titles=[
        'l={}<br> m={}'.format(ii, jj - ii) if jj < 2*ii+1 else ""
        for ii in range(nrows)
        for jj in range(ncols)
    ]
)

for l in range(LMAX+1):
    for m in range(-l, l+1):
        
        r0 = np.abs(SH[l][:,:, m+l])
        r0 /= r0.max()
        p0 = np.angle(SH[l][:,:, m+l])

        fig.add_trace(
            go.Surface(
                x=r0*x0, y=r0*y0, z=z0*r0,
                surfacecolor=p0,
                showscale=False,
                colorscale='PiYG',
                hoverinfo='none',
            ),
            row=l+1, col=m+l+1,
        )

for ii in range(nrows):
    for jj in range(ncols):
        fig['layout']['scene{}'.format(ii*ncols + jj + 1)].update(
            dict(
                # xaxis_title='',
                # yaxis_title='',
                # zaxis_title='',
                xaxis_showticklabels=False,
                yaxis_showticklabels=False,
                zaxis_showticklabels=False,
                # xaxis = dict(range=[-1, 1]),
                # yaxis = dict(range=[-1, 1]),
                # zaxis = dict(range=[-1, 1]),
                # xaxis_visible=False,
                # yaxis_visible=False,
                # zaxis_visible=False
            )
        )

fig.update_layout(
    width=640, height=640,
)
fig.update_annotations(
    font_size=10
)

fig.write_html('spherical_harmonics.html', include_plotlyjs=False, full_html=False)  
fig.show()
