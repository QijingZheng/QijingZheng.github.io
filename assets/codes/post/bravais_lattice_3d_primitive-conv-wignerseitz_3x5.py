#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import Delaunay

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ase.lattice import all_variants

############################################################

def get_WignerSeitz_3d(cell):
    """
    Constructed the Wigner-Seitz cell of a given lattice by Voronoi
    decomposition.  A Voronoi diagram is a subdivision of the space into the
    nearest neighborhoods of a given set of points. 

    https://en.wikipedia.org/wiki/Wigner%E2%80%93Seitz_cell
    https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html#voronoi-diagrams
    """

    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    from scipy.spatial import Voronoi
    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    # for rid in vor.ridge_vertices:
    #     if( np.all(np.array(rid) >= 0) ):
    #         bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
    #         bz_facets.append(vor.vertices[rid])

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        # WHY 13 ????
        # The Voronoi ridges/facets are perpendicular to the lines drawn between the
        # input points. The 14th input point is [0, 0, 0].
        if(pid[0] == 13 or pid[1] == 13):
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return vor.vertices[bz_vertices], bz_ridges, bz_facets


def get_primitive_cell_3d(cell):
    """
    Get the vertices, edges and facets of the primitive cell.
    """
    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    dx, dy, dz = np.mgrid[0:2, 0:2, 0:2]
    dxyz = np.c_[dx.ravel(), dy.ravel(), dz.ravel()]
    px, py, pz = np.tensordot(cell, [dx, dy, dz], axes=[0, 0])
    vertices = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    edges = []
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 5, 7, 3],
        [3, 7, 6, 2],
        [2, 6, 4, 0],
    ]

    for ii in range(len(vertices)):
        for jj in range(ii):
            if np.abs(dxyz[ii] - dxyz[jj]).sum() == 1:
                edges.append(np.vstack([vertices[ii], vertices[jj]]))

    return vertices, edges, [vertices[f] for f in faces]


bravais_latt_3d = {
    b.variant : (b.longname, b) for b in all_variants()
}

bravais_latt_keys = [
    'CUB', 'BCC', 'FCC', 'TET', 'BCT1',
    'ORC', 'ORCI', 'ORCC', 'ORCF1', 'RHL1',
    'HEX', 'MCL', 'MCLC1', 'TRI1a'
]
bravais_latt_des = [
    # bravais_latt_3d[bravais_latt_keys[ii]][0] for ii in range(14)
    r"Primitive Cubic<br>",
    r"Body-centered <br>Cubic (bcc)<br>",
    r"Face-centered <br>Cubic (fcc)<br>",
    r"Primitive Tetragonal<br>",
    r"Body-centered <br>Tetragonal<br>",
    r"Primitive<br>Orthorhombic<br>",
    r"Body-centered<br>Orthorhombic<br>",
    r"Base-centered<br>Orthorhombic<br>",
    r"Face-centered<br>Orthorhombic<br>",
    r"Rhombohedral<br>",
    r"Hexagonal<br>",
    r"Simple<br>Monoclinic<br>",
    r"Base-centered<br>Monoclinic<br>",
    r"Triclinic<br>",
]

############################################################
if __name__ == "__main__":
    
    ############################################################
    nrows = 3
    ncols = 5

    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[
            [{'is_3d': True} for jj in range(ncols)]
            for ii in range(nrows)
        ],
        subplot_titles = bravais_latt_des,
        horizontal_spacing = 0.02,
        vertical_spacing = 0.04,
    )
    ############################################################
    for irow in range(nrows):
        for icol in range(ncols):
            ilatt = irow * ncols + icol
            if ilatt >= 14:
                break
            bkey  = bravais_latt_keys[ilatt]
            blatt = bravais_latt_3d[bkey]

            pcell = blatt[1].tocell().array
            ccell = blatt[1].conventional().tocell().array

            all_cells = [pcell, ccell]
            
            all_cell_info = [
                get_primitive_cell_3d(pcell),
                get_primitive_cell_3d(ccell),
                get_WignerSeitz_3d(pcell),
            ]
            cell_clrs = ['red', 'blue', 'green']

            for icell, info in enumerate(all_cell_info):
                if icell == 1 and np.allclose(pcell, ccell):
                    continue

                if icell < 2:
                    basis_labs = [r'{}<sub>{}</sub>'.format('ab'[icell], ii+1) for ii in range(3)]
                    for ii, basis in enumerate(all_cells[icell]):
                        bx, by, bz = basis
                        fig.add_trace(
                            go.Scatter3d(
                                x=[0, bx],
                                y=[0, by],
                                z=[0, bz],
                                opacity=0.8,
                                hoverinfo='skip',
                                mode='text',
                                text=['0', basis_labs[ii]],
                                textfont=dict(color=cell_clrs[icell], size=12),
                            ),
                            row=irow+1, col=icol+1,
                        )

                Verts, Edges, Facets = info

                # the vertices
                fig.add_trace(
                    go.Scatter3d(
                        x=Verts[:,0],
                        y=Verts[:,1],
                        z=Verts[:,2],
                        opacity=0.8,
                        hoverinfo='skip',
                        mode='markers',
                        marker=dict(
                            # color=cell_clrs[icell],
                            color='black',
                            size=3,
                        ),
                    ),
                    row = irow+1, col=icol+1
                )
        
                # the edges
                for l in Edges:
                    x, y, z = l.T
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=x, y=y, z=z,
                            opacity=0.8,
                            hoverinfo='skip',
                            mode='lines',
                            line=dict(
                                color=cell_clrs[icell],
                                width=3,
                            ),
                        ),
                        row = irow+1, col=icol+1
                    )
        
                # the facets
                for ff in Facets:
                    # Add another point outside the facets so that the Delaunay works
                    # properly
                    if icell != 2:
                        simplex_g = np.vstack([np.average(all_cells[icell], axis=0), ff])
                    else:
                        simplex_g = np.vstack([[0, 0, 0], ff])
                    tri = Delaunay(simplex_g)
                    for ii, xx in enumerate(tri.simplices):
                        # exclude the extra point when plotting faces
                        x, y, z = simplex_g[xx[xx!=0]].T
                        fig.add_trace(
                            go.Mesh3d(
                                x=x, y=y, z=z,
                                opacity=0.1,
                                hoverinfo='skip',
                                color=cell_clrs[icell],
                                i=[0], j=[1], k=[2],
                            ),
                            row=irow+1, col=icol+1,
                        )

    
    ############################################################
    # camera = dict(
    #   up=dict(x=0, y=0, z=1),
    #   center=dict(x=0, y=0, z=0),
    #   eye=dict(x=1.00, y=-1.20, z=0.00)
    # )
    margin=dict(l=0, r=0, b=0)
    SCENES = {}
    for ii in range(14):
        SCENES['scene{}'.format(ii+1)] =  dict(
            # camera=camera,
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

        fig.layout.annotations[ii].update(
            # y=0,
            # yref="y{} domain".format(ii+2),
            yanchor='bottom',
            font={
                "size": 12,
                # "weight": "bold",
            }
        )
    
    fig.update_layout(
        width=720, height=720,
        margin=margin,
        showlegend=False,
        **SCENES,
        # title = {
        #     "text": "Primitive and Wigner-Seitz Cell",
        #     "font": {'size':24},
        #     "x": 0.5,
        # }
    )
    # fig.update_scenes(camera_projection_type='orthographic')
    # fix the ratio in the top left subplot to be a cube
    fig.update_layout(scene_aspectmode='cube')
    
    fig.write_html('blatt_prim_conv_ws_cell_3x5.html', include_plotlyjs=False, full_html=False)  
    fig.show()
    ############################################################
