#!/usr/bin/env python

import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()
fig.update_layout(
    width=720, height=600,
    title="",
    yaxis=dict(
        title=r'$\omega \thinspace/ \thinspace\omega_{\text{max}}$',
        titlefont=dict(size=20)
    ),
    xaxis = dict(
        tickvals=[-np.pi, 0, np.pi],
        ticktext=[r'$\mbox{-}{\pi\over a}$', r'$0$', r'$\pi\over a$'],
        tickfont=dict(size=20)
    ),
    legend=dict(
        font=dict(size=20)
    )
)

x0 = np.linspace(-np.pi, np.pi, 300)
mass_ratio = np.linspace(0, 1, 21)

# Add traces, one for each slider step
for m0 in mass_ratio:
    # optical mode
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#FF0000", width=3),
            name=r'$\omega_{+}$',
            x=x0,
            y=1/np.sqrt(2)*np.sqrt(1 + np.sqrt(
                (1 + m0**2 + 2 * m0 * np.cos(x0)) / (1 + m0)**2
            ))
        )
    )
    # acoustic mode
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#0000FF", width=3),
            name=r'$\omega_{-}$',
            x=x0,
            y=1/np.sqrt(2)*np.sqrt(1 - np.sqrt(
                (1 + m0**2 + 2 * m0 * np.cos(x0)) / (1 + m0)**2
            ))
        )
    )

# Make 10th trace visible
fig.data[10].visible = True
fig.data[11].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data))[::2]:
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)}],  # layout attribute
        label='{:.2f}'.format(mass_ratio[i//2])
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": 'm/M = '},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
)

fig.write_html('1d_diatomic_chain.html', include_plotlyjs=False, full_html=False)
fig.show()
