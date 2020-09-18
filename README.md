# NeRF
A Numpy Implementation of the NeRF Algoritm for Global and Internal Molecular Coordinate Conversion

### Installation
`pip install pynerf`

### Format
```python
import numpy as np

# Global coordinates for the molecule: CCCCC(C)CCCc1ccccc1
# X, Y, Z
XYZ = np.array([
    [-3.85918113,  1.96727702, -0.90964251],
    [-3.18010648,  0.73561076, -1.50023432],
    [-2.59821482, -0.0276012 , -0.32047649],
    [-1.89223409, -1.27462464, -0.74157415],
    [-1.36589543, -1.96407604,  0.4887215 ],
    [-2.5080429 , -2.32882212,  1.43590953],
    [-0.42564473, -1.07397927,  1.26079092],
    [ 0.74822042, -0.69585885,  0.37285949],
    [ 1.64031545,  0.1825755 ,  1.22863933],
    [ 2.84575382,  0.63093342,  0.47241165],
    [ 2.86522176,  1.79613148, -0.26713038],
    [ 3.99177109,  2.19435562, -0.96077995],
    [ 5.14810842,  1.44285119, -0.9439963 ],
    [ 5.1362438 ,  0.27652072, -0.20673226],
    [ 4.0076203 , -0.12008426,  0.48671673]
])

# Internal coordinates for the molecule: CCCCC(C)CCC
# BondLength, BondAngle (Deg/Rad), BondTorsion (Deg/Rad)
DOF = np.array([
    [0.000,   0.000,    0.000],
    [1.525,   0.000,    0.000],
    [1.521, 105.973,    0.000],
    [1.494, 112.404, -180.000],
    [1.505, 108.496, -179.436],
    [1.528, 110.770,   60.154],
    [1.507, 111.484, -118.831],
    [1.520, 109.236,  -59.789],
    [1.517, 105.593, -179.530],
    [1.492, 111.304,  179.999],
    [1.380, 122.458,   89.997],
    [1.382, 121.590,  179.999],
    [1.379, 121.364,    0.000],
    [1.380, 117.378,    0.000],
    [1.383, 121.159,    0.000],
])

# Custom dependencies with branch
# PrevAtom1, PrevAtom2, PrevAtom3
DEP = np.array([
    [ 0,  0,  0],
    [ 0,  0,  0],
    [ 1,  0,  0],
    [ 2,  1,  0],
    [ 3,  2,  1],
    [ 4,  3,  2],
    [ 4,  3,  5], # <- Branch Point
    [ 6,  4,  3],
    [ 7,  6,  4],
    [ 8,  7,  6],
    [ 9,  8,  7],
    [10,  9,  8],
    [11, 10,  9],
    [12, 11, 10],
    [13, 12, 11]
])

# Note: Default assumes all atoms are sequential

BONDS = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
]
```

### Example
```python
from nerf import NeRF, iNeRF

xyz = NeRF(DOF, deps=DEP)
dof = iNeRF(xyz, deps=DEP)

assert np.all(np.absolute(xyz - XYZ) < 0.001)
assert np.all(np.absolute(dof - DOF) < 0.001)
```

### Vectorized Example
```python
from nerf import NeRF, iNeRF, perturb_dofs

repeats = 100000 # ~2 seconds on my computer for both calculations

DOFS = perturb_dofs(
    np.repeat(DOF[np.newaxis], repeats, axis=0),
    bond_length_factor=0.01 * np.ones(1),
    bond_angle_factor=0.1 * np.ones(len(DOF)),
    bond_torsion_factor=1.0 * np.ones((repeats,len(DOF)))
)

xyzs = NeRF(DOFS, deps=DEP)
dofs = iNeRF(xyzs, deps=DEP)

xyzs_delta = xyzs - ORIGIN_XYZ

dofs_delta = dofs - DOF
dofs_delta[np.where(dofs_delta > 180.0)] -= 360.0
dofs_delta[np.where(dofs_delta < -180.0)] += 360.0

assert np.all(np.absolute(np.mean(xyzs_delta, axis=0)) < 0.05)
assert np.all(np.absolute(np.mean(dofs_delta, axis=0)) < 0.05)
```

### Deps Example
```python
assert np.array_equal(build_deps(BONDS), DEP)
```

### Citation
Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE. Practical conversion from torsion space to Cartesian space for in silico protein synthesis. J Comput Chem. 2005;26(10):1063-1068. doi:10.1002/jcc.20237
