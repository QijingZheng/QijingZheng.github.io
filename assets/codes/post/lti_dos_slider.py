#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from itertools import combinations

if __name__ == "__main__":
    ############################################################
    points_tet = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ], dtype=float)
    # the value at the vertices
    vals_tet       = np.array([0, 1, 2, 3])
    e1, e2, e3, e4 = vals_tet
    e21, e31, e41  = vals_tet[1:] - vals_tet[0]

    
    ############################################################
    fig = go.Figure()
    ############################################################
    # the vertex
    fig.add_trace(
        go.Scatter3d(
            # 4 vertices of a tetrahedron
            x=points_tet[:,0],
            y=points_tet[:,1],
            z=points_tet[:,2],
            opacity=0.8,
            hoverinfo='skip',
            mode='markers+text',
            marker=dict(
                color='black',
                size=5,
            ),
            text=["ε<sub>{}</sub>".format(ii+1) for ii in range(4)],
            textfont=dict(color='blue', size=18)
        )
    )
    # the edge
    for ii in combinations(range(4), 2):
        x, y, z = points_tet[list(ii)].T
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                line=dict(
                    color='black',
                    width=2,
                ),
                hoverinfo='skip',
                mode='lines',
            )
        )

    ndata1  = len(fig.data)

    energies = np.r_[
        np.linspace(e1, e2, 5, endpoint=False),
        np.linspace(e2, e3, 5, endpoint=False),
        np.linspace(e3, e4, 5, endpoint=True),
    ]
    nslider = len(energies)
    vert_energy_idx = [0, 5, 10, 14]

    for ii, e in enumerate(energies):
        cross_pts = None
        cpts_conn = None
        e01 = e - e1

        if e1 <= e <= e2:
            cross_pts = np.array([
                [e01 / e21, 0, 0],
                [0, e01 / e31, 0],
                [0, 0, e01 / e41],
            ])
            cpts_conn = [[0], [1], [2]]
        elif e2 < e <= e3:
            cross_pts = np.array([
                [(e01 - e41) / (e21 - e41), 0, (e21 - e01) / (e21 - e41)],
                [(e01 - e31) / (e21 - e31), (e21 - e01) / (e21 - e31), 0],
                [0, e01 / e31, 0],
                [0, 0, e01 / e41],
            ])
            cpts_conn = [[0, 1], [1, 2], [3, 3]]
        elif e3 < e <= e4:
            cross_pts = np.array([
                [(e01 - e41) / (e21 - e41), 0, (e21 - e01) / (e21 - e41)],
                [0, (e01 - e41) / (e31 - e41), (e31 - e01) / (e31 - e41)],
                [0, 0, e01 / e41],
            ])
            cpts_conn = [[0], [1], [2]]
        else:
            pass

        if ii == 3:
            visible=True
        else:
            visible=False

        if cross_pts is not None:
            x, y, z = cross_pts.T
            i, j, k = cpts_conn
            fig.add_trace(
                go.Mesh3d(
                    visible=visible,
                    x=x, y=y, z=z,
                    opacity=0.8,
                    hoverinfo='skip',
                    color='red',
                    # # i, j and k give the vertices of triangles
                    i=i, j=j, k=k,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    # 4 vertices of a tetrahedron
                    x=x, y=y, z=z,
                    visible=visible,
                    opacity=0.8,
                    hoverinfo='skip',
                    mode='text',
                    text=["c<sub>{}</sub>".format(ii+1) for ii in range(len(cross_pts))],
                    textfont=dict(color='red', size=18)
                )
            )

    ############################################################
    ndata2 = len(fig.data) - ndata1
    steps = []
    for ii in range(nslider):
        label = ''
        if ii in vert_energy_idx:
            idx = vert_energy_idx.index(ii)
            label = "ε<sub>{}</sub>".format(idx+1)
            print(label)
        step = dict(
            method="update",
            args=[
                {"visible": [True] * ndata1 + [False] * ndata2},
                # {"title": "Slider switched to step: " + str(ii)}
            ],  # layout attribute
            label=label,
        )
        for jj in range(2):
            step["args"][0]["visible"][
                ndata1 + 2 * ii + jj
            ] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=3,
        currentvalue={'prefix': 'Disp:'},
        # len=0.50,
        # x=0.0,
        xanchor='left',
        pad={"b": 20, 't': 0},
        steps=steps,
    )]

    fig.update_layout(
        sliders=sliders
    )

    ############################################################
    # delta = 0.3
    camera = dict(
	up=dict(x=0, y=0, z=1),
	center=dict(x=0, y=0, z=0),
	eye=dict(x=1.50, y=-1.20, z=0.00)
    )
    scene = dict(
        camera=camera,
        xaxis_showbackground=False,
        yaxis_showbackground=False,
        zaxis_showbackground=False,
        xaxis_title='',
        yaxis_title='',
        zaxis_title='',
        xaxis_tickvals=[],
        yaxis_tickvals=[],
        zaxis_tickvals=[],
    )
    margin=dict(l=0, r=0, t=20, b=20)

    fig.update_layout(
        width=480, height=480,
        margin=margin,
        showlegend=False,
        scene=scene,
    )

    ############################################################
    # fig.update_scenes(camera_projection_type='orthographic')
    # fix the ratio in the top left subplot to be a cube
    fig.update_layout(scene_aspectmode='cube')

    fig.write_html('tet1.html', include_plotlyjs=False, full_html=False)  
    fig.show()
