#!/usr/bin/env python

import numpy as np
import plotly.graph_objects as go

############################################################
N = 101
t = 1.0
a = 1.0

kx, ky = np.mgrid[-3:3:1j*N, -3:3:1j*N]
E1 = t * np.sqrt(
    3 +
    2 * np.cos(np.sqrt(3)*ky*a) +
    4 * np.cos(3/2.*kx*a)*np.cos(np.sqrt(3)/2*ky*a)
)
E2 = -E1

############################################################

fig = go.Figure(
    data=[
        go.Surface(
            z=E1, x=kx, y=ky,
            colorscale='Reds', showscale=False, opacity=1.0,
            hoverinfo='none'
        ),
        go.Surface(
            z=E2, x=kx, y=ky,
            colorscale='Blues', showscale=False, opacity=1.0,
            hoverinfo='none'
        )
    ],
)

fig.update_layout(
    width=600, height=600,
    scene=dict(
        xaxis_title='kx',
        yaxis_title='ky',
        zaxis_title='E(kx, ky)',
    )
)

fig.write_html('graphene_tb_band.html', include_plotlyjs=False, full_html=False)  
fig.show()
