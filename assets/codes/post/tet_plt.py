#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from scipy.spatial import Delaunay

def tetrahedron_facets(pts):
    '''
    '''
    from itertools import combinations

    npts = len(pts)
    return [pts[list(x)] for x in combinations(range(npts), 3)]
    
def tetrahedron_edges(pts):
    '''
    '''
    from itertools import combinations

    npts = len(pts)
    return [pts[list(x)] for x in combinations(range(npts), 2)]

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
    fig = plt.figure(
        figsize=plt.figaspect(0.30),
        dpi=300,
    )
    axes = [
        plt.subplot(1, 3, ii+1, projection='3d') for ii in range(3)
    ]
    ############################################################
    ax = axes[0]
    # ax.scatter(
    #     points[:, 0],
    #     points[:, 1],
    #     points[:, 2],
    #     marker='o',
    #     s=100, c='r', lw=0.0,
    # )

    edges = Line3DCollection(
        [[points[ii], points[jj]] for ii in range(8) for jj in range(ii) if
            np.linalg.norm(points[ii] - points[jj]) < 1.01],
        lw=2.0,
        color='k',
        alpha=0.5,
    )

    ax.add_collection3d(edges)

    # clrs = np.repeat(
    #    plt.colormaps['jet'](
    #        np.linspace(0, 1, len(tet.simplices))
    #    ),
    #    4, axis=0
    # )
    # vts = [
    #     x for tt in tet.simplices
    #       for x in tetrahedron_facets(points[tt])
    # ]
    # tri = Poly3DCollection(
    #     vts,
    #     facecolors=clrs,
    #     alpha=0.5,
    # )
    # b3d = Line3DCollection(
    #     vts,
    #     lw=2.0,
    #     colors=clrs,
    # )

    ############################################################
    ax = axes[1]

    clrs = plt.colormaps['jet'](
       np.linspace(0, 1, len(tet.simplices))
    )
    for ii, tt in enumerate(tet.simplices):
        edges = Line3DCollection(
            [xx for xx in tetrahedron_edges(points[tt])],
            lw=1.0, ls='--',
            color='b',
            alpha=0.5,
        )
        ax.add_collection3d(edges)

        faces = Poly3DCollection(
            [xx for xx in tetrahedron_facets(points[tt])],
            facecolor=clrs[ii],
            alpha=0.5,
        )
        # ax.add_collection3d(faces)
    ############################################################
    ax = axes[2]

    # ax.scatter(
    #     points[:, 0],
    #     points[:, 1],
    #     points[:, 2],
    #     marker='o',
    #     s=100, c='r', lw=0.0,
    #
    # )
    # edges = Line3DCollection(
    #     [[points[ii], points[jj]] for ii in range(8) for jj in range(ii) if
    #         np.linalg.norm(points[ii] - points[jj]) < 1.01],
    #     lw=1.0,
    #     ls='--',
    #     color='k',
    #     alpha=0.4,
    # )
    # ax.add_collection3d(edges)

    clrs = plt.colormaps['jet_r'](np.linspace(0, 1, len(tet.simplices)))
    for ii, tt in enumerate(tet.simplices):
        mid2 = np.average(
            points[[jj for jj in tt if jj not in diag_pts_idx]], axis=0)
        exp_dir = mid2 - diag_mid
        exp_dir /= np.linalg.norm(exp_dir)
        
        faces = Poly3DCollection(
            [xx + 0.3 * exp_dir for xx in tetrahedron_facets(points[tt])],
            facecolor=clrs[ii],
            alpha=0.5,
        )

        edges = Line3DCollection(
            [xx + 0.3 * exp_dir for xx in tetrahedron_edges(points[tt])],
            lw=2.0,
            color='k',
            alpha=0.5,
        )

        ax.add_collection3d(faces)
        ax.add_collection3d(edges)


    ############################################################
    for ax in axes:
        delta = 0.2
        ax.set_xlim(-delta, 1 + delta)
        ax.set_ylim(-delta, 1 + delta)
        ax.set_zlim(-delta, 1 + delta)

        ax.view_init(elev=15, azim=120)
        ax.axis('off')
    ############################################################
    plt.tight_layout(pad=0)
    plt.savefig('tet.png', dpi=480)
    # plt.show()

    from subprocess import call
    call('feh -xdF tet.png'.split())
    
