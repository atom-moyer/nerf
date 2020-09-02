# NeRF
A Numpy Implementation of the NeRF Algoritm for Global and Internal Molecular Coordinate Conversion

### Installation
`pip install pynerf`

### Format
```python
import numpy as np

# Z-matrix for the molecule: CCCCCC(C)CC
# BondLength, BondAngle (Deg/Rad), BondTorsion (Deg/Rad)
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

# Global coordinates for the molecule: CCCCCC(C)CC
# X, Y, Z
XYZ = np.array([
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

# Custom dependencies with branch
# PrevAtom1, PrevAtom2, PrevAtom3
DEP = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [2, 1, 0],
    [3, 2, 1],
    [4, 3, 2],
    [4, 3, 5], # <- Branch Point
    [6, 4, 3],
    [7, 6, 4],
])
```

### Example
```python
from nerf import NeRF, iNeRF

xyz = NeRF(DOF, dependency=DEP)
dof = iNeRF(xyz, dependency=DEP)

assert np.all(np.absolute(xyz - XYZ) < 0.001)
assert np.all(np.absolute(dof - DOF) < 0.001)
```

### Vectorized Example
```python
from nerf import NeRF, iNeRF, perturb_dofs

DOFS = perturb_dofs(np.repeat(DOF[np.newaxis], repeats, axis=0))

xyzs = NeRF(DOFS, dependency=DEP)
dofs = iNeRF(xyzs, dependency=DEP)

assert np.all(np.absolute(np.mean(xyzs, axis=0) - XYZ) < 0.001)
assert np.all(np.absolute(np.mean(dofs, axis=0) - DOF) < 0.001)
```

### Citation
Parsons J, Holmes JB, Rojas JM, Tsai J, Strauss CE. Practical conversion from torsion space to Cartesian space for in silico protein synthesis. J Comput Chem. 2005;26(10):1063-1068. doi:10.1002/jcc.20237
