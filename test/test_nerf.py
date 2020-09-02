import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

from nerf import NeRF, iNeRF, perturb_dofs


repeats = 1000000


XYZ = np.array([
    [ 3.969, -1.316, -0.339],
    [ 3.149, -0.051, -0.114],
    [ 1.659, -0.354,  0.031],
    [ 0.841,  0.921,  0.230],
    [-0.627,  0.679,  0.606],
    [-1.264,  2.002,  1.035],
    [-1.400,  0.041, -0.553],
    [-2.831, -0.362, -0.196],
    [-3.475, -1.176, -1.312]
])


ORIGIN_XYZ = np.array([
    [ 0.000,  0.000,  0.000],
    [ 1.524,  0.000,  0.000],
    [ 2.096,  1.416,  0.000],
    [ 3.623,  1.408, -0.026],
    [ 4.268,  2.782,  0.198],
    [ 5.772,  2.607,  0.415],
    [ 3.983,  3.728, -0.973],
    [ 4.471,  5.160, -0.746],
    [ 3.977,  6.096, -1.843]
])


DOF = np.array([
    [0.000,   0.000,    0.000],
    [1.524,   0.000,    0.000],
    [1.527, 111.974,    0.000],
    [1.528, 111.660, -178.932],
    [1.535, 114.360, -170.687],
    [1.530, 109.309,  169.305],
    [1.532, 111.280,  123.249],
    [1.529, 113.910,  173.894],
    [1.524, 111.425, -171.369]
])


DEP = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [2, 1, 0],
    [3, 2, 1],
    [4, 3, 2],
    [4, 3, 5],
    [6, 4, 3],
    [7, 6, 4],
])


def test_nerf():
    xyz = NeRF(DOF, dependency=DEP)
    dof = iNeRF(xyz, dependency=DEP)

    assert np.all(np.absolute(xyz - ORIGIN_XYZ) < 0.001)
    assert np.all(np.absolute(dof - DOF) < 0.001)

def test_nerf_vectorized():
    DOFS = perturb_dofs(np.repeat(DOF[np.newaxis], repeats, axis=0))

    xyzs = NeRF(DOFS, dependency=DEP)
    dofs = iNeRF(xyzs, dependency=DEP)

    assert np.all(np.absolute(np.mean(xyzs, axis=0) - ORIGIN_XYZ) < 0.001)
    assert np.all(np.absolute(np.mean(dofs, axis=0) - DOF) < 0.001)
