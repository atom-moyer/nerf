import numpy as np


def norm(a):
    return np.linalg.norm(a, axis=-1, keepdims=True)


def normalize(a):
    a /= norm(a)
    return a


def dot(a, b):
    return np.sum(a*b, axis=-1, keepdims=True)


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


def NeRF(dofs, dependency=None, degrees=True):
    xyzs = np.empty_like(dofs)

    locals = sph2cart(dofs, degrees)

    for i in range(dofs.shape[-2]):
        if i == 0:
            xyzs[...,0,0] = 0.0
            xyzs[...,0,1] = 0.0
            xyzs[...,0,2] = 0.0
        elif i == 1:
            xyzs[...,1,0] = dofs[...,1,0]
            xyzs[...,1,1] = 0.0
            xyzs[...,1,2] = 0.0
        elif i == 2:
            xyzs[...,2,0] = dofs[...,2,0] * np.cos(np.pi - np.radians(dofs[...,2,1])) + dofs[...,1,0]
            xyzs[...,2,1] = dofs[...,2,0] * np.sin(np.pi - np.radians(dofs[...,2,1]))
            xyzs[...,2,2] = 0.0
        else:
            if dependency is None:
                a = xyzs[...,i-3,:]
                b = xyzs[...,i-2,:]
                c = xyzs[...,i-1,:]
            else:
                a = xyzs[...,dependency[i,2],:]
                b = xyzs[...,dependency[i,1],:]
                c = xyzs[...,dependency[i,0],:]

            ab = normalize(b - a)
            bc = normalize(c - b)

            n = normalize(np.cross(ab, bc))
            n_x_bc = normalize(np.cross(n, bc))

            M = np.stack([bc, n_x_bc, n], axis=-1)

            xyzs[...,i,:] = np.squeeze(M @ np.swapaxes(locals[...,[i],:], -1, -2)) + c

    return xyzs


def iNeRF(xyzs, dependency=None, degrees=True):
    dofs = np.empty_like(xyzs)

    for i in range(xyzs.shape[-2]):
        if i == 0:
            dofs[...,0,[0]] = 0.0
            dofs[...,0,[1]] = 0.0
            dofs[...,0,[2]] = 0.0
        elif i == 1:
            a = xyzs[...,0,:]
            b = xyzs[...,1,:]

            dofs[...,1,[0]] = norm(b - a)
            dofs[...,1,[1]] = 0.0
            dofs[...,1,[2]] = 0.0
        elif i == 2:
            a = xyzs[...,0,:]
            b = xyzs[...,1,:]
            c = xyzs[...,2,:]

            ab = normalize(a - b)
            bc = normalize(b - c)

            x = np.clip(dot(ab, bc), -1, 1)

            dofs[...,2,[0]] = norm(b - c)
            dofs[...,2,[1]] = np.pi - np.arccos(x)
            dofs[...,2,[2]] = 0.0
        else:
            if dependency is None:
                a = xyzs[...,i-3,:]
                b = xyzs[...,i-2,:]
                c = xyzs[...,i-1,:]
                d = xyzs[...,i,:]
            else:
                a = xyzs[...,dependency[i,2],:]
                b = xyzs[...,dependency[i,1],:]
                c = xyzs[...,dependency[i,0],:]
                d = xyzs[...,i,:]

            ba = normalize(b - a)
            bc = normalize(b - c)
            cd = normalize(c - d)

            v = ba - dot(ba, bc) * bc
            w = cd - dot(cd, bc) * bc

            x = np.clip(dot(bc, cd), -1, 1)
            y = np.clip(dot(v, w), -1, 1)
            z = np.clip(dot(np.cross(bc, v), w), -1, 1)

            dofs[...,i,[0]] = norm(d - c)
            dofs[...,i,[1]] = np.pi - np.arccos(x)
            dofs[...,i,[2]] = -np.arctan2(z, y)

    if degrees:
        dofs[...,:,[1,2]] = np.degrees(dofs[...,:,[1,2]])

    return dofs


def perturb_dofs(dofs, bond_length_factor=0.01, bond_angle_factor=0.1, bond_torsion_factor=1.0):
    """
    dofs - [N,M,3]
    xyzs - [N,M,3]
    mapping - [2,<=M]
    """

    dofs[...,:,0] += np.random.uniform(-1, 1, dofs.shape[:-1]) * bond_length_factor
    dofs[...,:,1] += np.random.uniform(-1, 1, dofs.shape[:-1]) * bond_angle_factor
    dofs[...,:,2] += np.random.uniform(-1, 1, dofs.shape[:-1]) * bond_torsion_factor

    return np.tril(dofs, k=-1)


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


# def NeRF(abcs, dofs, degrees=True):
#     assert abcs.shape[:-2] == dofs.shape[:-2]
#     assert abcs.shape[-2] == 3 and abcs.shape[-1] == 3 and dofs.shape[-1] == 3
#
#     xyzs = sph2cart(dofs, degrees)
#
#     a = abcs[...,0,:]
#     b = abcs[...,1,:]
#     c = abcs[...,2,:]
#
#     ba = normalize(b - a)
#     cb = normalize(c - b)
#
#     for i in range(xyzs.shape[-2]):
#         n = normalize(np.cross(ba, cb))
#         n_x_cb = normalize(np.cross(n, cb))
#         M = np.stack([cb, n_x_cb, n], axis=-1)
#
#         xyzs[...,i,:] = c + np.squeeze(M @ np.swapaxes(xyzs[...,[i],:], -1, -2))
#
#         a = b
#         b = c
#         c = xyzs[...,i,:]
#
#         ba = cb
#         cb = normalize(c - b)
#
#     return xyzs
#
#
# def iNeRF(abcs, xyzs, degrees=True):
#     assert abcs.shape[:-2] == xyzs.shape[:-2]
#     assert abcs.shape[-2] == 3 and abcs.shape[-1] == 3 and xyzs.shape[-1] == 3
#
#     dofs = np.empty_like(xyzs)
#
#     a = abcs[...,0,:]
#     b = abcs[...,1,:]
#     c = abcs[...,2,:]
#
#     for i in range(dofs.shape[-2]):
#         d = xyzs[...,i,:]
#
#         ba = normalize(b - a)
#         bc = normalize(b - c)
#         cd = normalize(c - d)
#
#         v = ba - dot(ba, bc) * bc
#         w = cd - dot(cd, bc) * bc
#
#         x = np.clip(dot(bc, cd), -1, 1)
#         y = np.clip(dot(v, w), -1, 1)
#         z = np.clip(dot(np.cross(bc, v), w), -1, 1)
#
#         dofs[...,i,0] = np.squeeze(norm(d - c))
#         dofs[...,i,1] = np.squeeze(np.pi - np.arccos(x))
#         dofs[...,i,2] = np.squeeze(-np.arctan2(z, y))
#
#         a = b
#         b = c
#         c = d
#
#     if degrees:
#         dofs[...,:,[1,2]] = np.degrees(dofs[...,:,[1,2]])
#
#     return dofs
