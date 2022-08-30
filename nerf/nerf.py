import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt

import numba


def normalize(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def dot(a, b):
    return np.sum(a*b, axis=-1, keepdims=True)


# @numba.njit
def sph2cart(spherical, degrees):
    if degrees:
        pi_spherical1 = np.pi - np.radians(spherical[...,:,1])
        spherical2 = np.radians(spherical[...,:,2])
    else:
        pi_spherical1 = np.pi - spherical[...,:,1]
        spherical2 = spherical[...,:,2]

    R_cos_T = spherical[...,:,0] * np.cos(pi_spherical1)
    R_sin_T = spherical[...,:,0] * np.sin(pi_spherical1)

    cos_P = np.cos(spherical2)
    sin_P = np.sin(spherical2)

    return np.stack([R_cos_T, R_sin_T*cos_P, R_sin_T*sin_P], axis=-1)


# @numba.njit
def NeRF(dofs, abcs=None, deps=None, degrees=True):
    """
    dofs - [...,M,3]
    abcs - [...,3,3]
    deps - [M,3]

    xyzs - [...,M,3]
    """

    xyzs = np.zeros_like(dofs)

    if abcs is not None:
        xyzs[...,:3,:] = abcs

    locals = sph2cart(dofs, degrees)

    for i in range(dofs.shape[-2]):
        if i == 0:
            continue
        elif i == 1:
            if abcs is not None:
                continue
            elif deps is None:
                a = np.array([0.0, 1.0, 0.0])
                b = np.array([1.0, 0.0, 0.0])
                c = xyzs[...,i-1,:]
            else:
                a = np.array([0.0, 1.0, 0.0])
                b = np.array([1.0, 0.0, 0.0])
                c = xyzs[...,deps[i,0],:]
        elif i == 2:
            if abcs is not None:
                continue
            elif deps is None:
                a = xyzs[...,i-2,:] + np.array([0.0, 1.0, 0.0])
                b = xyzs[...,i-2,:]
                c = xyzs[...,i-1,:]
            else:
                a = xyzs[...,deps[i,1],:] + np.array([0.0, 1.0, 0.0])
                b = xyzs[...,deps[i,1],:]
                c = xyzs[...,deps[i,0],:]
        else:
            if deps is None:
                a = xyzs[...,i-3,:]
                b = xyzs[...,i-2,:]
                c = xyzs[...,i-1,:]
            else:
                a = xyzs[...,deps[i,2],:]
                b = xyzs[...,deps[i,1],:]
                c = xyzs[...,deps[i,0],:]

        ab = normalize(b - a)
        bc = normalize(c - b)

        n = normalize(np.cross(ab, bc))
        n_x_bc = normalize(np.cross(n, bc))

        M = np.stack([bc, n_x_bc, n], axis=-1)

        globals = np.squeeze(M @ np.swapaxes(locals[...,[i],:], -1, -2))

        xyzs[...,i,:] = globals + c

    return xyzs


# @numba.njit
def iNeRF(xyzs, deps=None, degrees=True):
    """
    xyzs - [...,M,3]
    deps - [M,3]

    dofs - [...,M,3]
    """

    dofs = np.zeros_like(xyzs)

    for i in range(xyzs.shape[-2]):
        if i == 0:
            dofs[...,i,0] = 0.0
            dofs[...,i,1] = 0.0
            dofs[...,i,2] = 0.0
        elif i == 1:
            if deps is None:
                c = xyzs[...,i-1,:]
            else:
                c = xyzs[...,deps[i,0],:]

            d = xyzs[...,i,:]

            dofs[...,i,0] = np.squeeze(np.linalg.norm(d-c, axis=-1, keepdims=True))
            dofs[...,i,1] = 0.0
            dofs[...,i,2] = 0.0

        elif i == 2:
            if deps is None:
                b = xyzs[...,i-2,:]
                c = xyzs[...,i-1,:]
            else:
                b = xyzs[...,deps[i,1],:]
                c = xyzs[...,deps[i,0],:]

            d = xyzs[...,i,:]

            bc = normalize(b - c)
            cd = normalize(c - d)

            x = np.clip(dot(bc, cd), -1.0, 1.0)

            dofs[...,i,0] = np.squeeze(np.linalg.norm(d-c, axis=-1, keepdims=True))
            dofs[...,i,1] = np.squeeze(np.pi - np.arccos(x))
            dofs[...,i,2] = 0.0
        else:
            if deps is None:
                a = xyzs[...,i-3,:]
                b = xyzs[...,i-2,:]
                c = xyzs[...,i-1,:]
            else:
                a = xyzs[...,deps[i,2],:]
                b = xyzs[...,deps[i,1],:]
                c = xyzs[...,deps[i,0],:]

            d = xyzs[...,i,:]

            ba = normalize(b - a)
            bc = normalize(b - c)
            cd = normalize(c - d)

            v = ba - dot(ba, bc) * bc
            w = cd - dot(cd, bc) * bc

            x = np.clip(dot(bc, cd), -1.0, 1.0)
            y = np.clip(dot(v, w), -1.0, 1.0)
            z = np.clip(dot(np.cross(bc, v), w), -1.0, 1.0)

            dofs[...,i,0] = np.squeeze(np.linalg.norm(d-c, axis=-1, keepdims=True))
            dofs[...,i,1] = np.squeeze(np.pi - np.arccos(x))
            dofs[...,i,2] = np.squeeze(-np.arctan2(z, y))

    if degrees:
        dofs[...,:,[1,2]] = np.degrees(dofs[...,:,[1,2]])

    return dofs


def perturb_dofs(dofs, bond_length_factor=0.01, bond_angle_factor=0.1, bond_torsion_factor=1.0, in_place=False):
    """
    dofs - [...,M,3]
    bond_length_factor - [...,M] broadcast-able
    bond_angle_factor - [...,M] broadcast-able
    bond_torsion_factor - [...,M] broadcast-able
    NOTE: Perturbs by (gaussian normal distribution * factor)

    dofs - [...,M,3] (copy)
    """
    if not in_place:
        dofs = np.copy(dofs)

    if bond_length_factor is not None:
        dofs[...,:,0] += np.random.normal(0, 1, dofs.shape[:-1]) * bond_length_factor

    if bond_angle_factor is not None:
        dofs[...,:,1] += np.random.normal(0, 1, dofs.shape[:-1]) * bond_angle_factor

    if bond_torsion_factor is not None:
        dofs[...,:,2] += np.random.normal(0, 1, dofs.shape[:-1]) * bond_torsion_factor

    return np.tril(dofs, k=-1)


def perturb_abcs(abcs, translation_factor=0.1, rotation_factor=3.0):
    """
    abcs - [...,3,3]
    translation_factor - [...,1,3] broadcast-able
    NOTE: Perturbs by (gaussian normal distribution * factor)

    dofs - [...,M,3] (copy)
    """
    abcs = np.copy(abcs)

    abcs[...,:,:] += np.random.normal(0, 1, (abcs.shape[0],1,3)) * translation_factor

    coms = np.mean(abcs, axis=-2, keepdims=True)

    rot_vecs = np.random.normal(0, 1, (abcs.shape[0],3))
    rot_vecs /= np.linalg.norm(rot_vecs, axis=-1, keepdims=True)
    rot_vecs *= np.radians(np.random.normal(0, 1, (abcs.shape[0],1)) * rotation_factor)

    rot_mats = sp.spatial.transform.Rotation.from_rotvec(rot_vecs).as_dcm()

    abcs -= coms
    abcs = abcs @ rot_mats
    abcs += coms

    return abcs


def build_deps(bonds, dump_png=None):
    """
    bonds - [M,M] (symmetric)

    deps - [M,3]
    """
    assert np.array_equal(bonds, np.transpose(bonds))

    parents = np.array([np.argwhere(bond)[-1] if np.any(bond) else -9999 for bond in np.tril(bonds, k=-1)], dtype=np.int)

    deps = []
    for i in np.arange(len(bonds)):
        # 1: Root Corrections
        if i == 0:
            par_i = 0
            gp_i = 0
            ggp_i = 0
        elif i == 1:
            par_i = 0
            gp_i = 0
            ggp_i = 0
        elif i == 2:
            par_i = 1
            gp_i = 0
            ggp_i = 0
        else:
            par_i = parents[i]
            gp_i = -9999 if (par_i==-9999) else parents[par_i]
            ggp_i = -9999 if (gp_i==-9999) else parents[gp_i]

        # Index 0 Branch Correction
        if (gp_i == -9999):
            gp_i = 1
            ggp_i = 2

        # Index 1 Branch Correction
        if (ggp_i == -9999):
            if (par_i == 1):
                ggp_i = 2
            else:
                ggp_i = 1

        deps.append([par_i, gp_i, ggp_i])

    # Clean
    deps = np.tril(deps, k=-1)

    # General Branch Correction
    for i, dep in enumerate(deps):
        matches = np.all(dep == deps[:i], axis=1)

        if np.any(matches):
            dep[2] = np.argwhere(matches)[0]

    if dump_png:
        graph = nx.from_numpy_matrix(bonds)
        nx.draw_spring(graph, with_labels=True)
        plt.savefig(dump_png)

    return deps


# ### Fast Hbond score
#
# ray_HO = acceptor_xyzs[i,0,:] - donor_xyzs[j,0,:]
# dist = normalize(ray_HO) - 2.00
#
# if dist < 0: dist *= 1.5
#
# if abs(dist) > 0.8:
#     continue
#
# dist_score = 1 - (dist/0.8)**2
#
# ray_CO = acceptor_xyzs[i,0,:] - acceptor_abcs[i,2,:]
# normalize(ray_CO)
#
# ray_NH = donor_xyzs[j,0,:] - donor_abcs[j,2,:]
# normalize(ray_NH)
#
# a_dirscore = ray_CO.dot(ray_HO) * -1
# h_dirscore = ray_NH.dot(ray_HO)
#
# if a_dirscore < 0 or h_dirscore < 0:
#     continue
#
# score += -1 * seq_dist_coeff * dist_score * dist_score * a_dirscore * h_dirscore * h_dirscore
