#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.spatial import Delaunay

if __name__ == "__main__":
    # Real Space Basis
    Acell = np.array([[0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])
    Acell = np.diag([1, 1, 1.])
    
    pframe = np.mgrid[0:2, 0:2, 0:2].reshape((3, -1)).T
    points = np.tensordot(
        Acell, np.mgrid[0:2, 0:2, 0:2], axes=(0,0)
    ).reshape(3, -1).T
    tet = Delaunay(points)

    up, uc       = np.unique(tet.simplices, return_counts=True)
    diag_pts_idx = up[np.argsort(uc)][-2:]
    diag_mid     = np.average(points[diag_pts_idx], axis=0)
    ############################################################
    nrows = 1
    ncols = 2

    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[
            [{'is_3d': True} for jj in range(ncols)]
            for ii in range(nrows)
        ],
    )
    ############################################################
    tet_clrs = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    verts_id = np.arange(8, dtype=int)
    for col in range(ncols):
        for ii, tt in enumerate(tet.simplices):
            if col == 0:
                x, y, z = points[tt].T
            else:
                mid2 = np.average(
                    points[[jj for jj in tt if jj not in diag_pts_idx]], axis=0)
                dd = mid2 - diag_mid
                dd /= np.linalg.norm(dd)
                x, y, z = (points[tt] + dd * 0.3).T

            fig.add_trace(
                go.Mesh3d(
                    # 4 vertices of a tetrahedron
                    x=x, y=y, z=z,
                    opacity=1.0,
                    color=tet_clrs[ii],
                    # # i, j and k give the vertices of triangles
                    i = [0, 0, 0, 1],
                    j = [1, 2, 3, 2],
                    k = [2, 3, 1, 3],
                ),
                row=1, col=col+1,
            )
            fig.add_trace(
                go.Scatter3d(
                    # 4 vertices of a tetrahedron
                    x=x, y=y, z=z,
                    opacity=0.8,
                    marker=dict(
                        color='black',
                        size=5,
                    )
                ),
                row=1, col=col+1,
            )

            if col == 1:
                fig.add_trace(
                    go.Scatter3d(
                        # 4 vertices of a tetrahedron
                        x=x, y=y, z=z,
                        opacity=0.8,
                        mode='text',
                        text=["{}".format(ii+1) for ii in verts_id[tt]],
                    ),
                    row=1, col=2,
                )


    x, y, z = points.T
    fig.add_trace(
        go.Scatter3d(
            # 4 vertices of a tetrahedron
            x=x, y=y, z=z,
            opacity=0.8,
            mode='text',
            text=["{}".format(ii+1) for ii in verts_id],
        ),
        row=1, col=1,
    )

    ############################################################
    # delta = 0.3
    camera = dict(
	up=dict(x=0, y=0, z=1),
	center=dict(x=0, y=0, z=0),
	eye=dict(x=-1.00, y=1.80, z=1.00)
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
        width=640, height=320,
        margin=margin,
        showlegend=False,
        scene=scene,
        scene2=scene,
    )

    # fig.update_scenes(camera_projection_type='orthographic')
    # fix the ratio in the top left subplot to be a cube
    fig.update_layout(scene_aspectmode='cube')

    fig.write_html('tet.html', include_plotlyjs=False, full_html=False)  
    fig.show()
