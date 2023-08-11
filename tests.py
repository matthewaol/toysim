from toysim import older
import numpy as np
import scipy as sp
from toysim.simulation import chunk_transform


def test_chunk_transform():
    test_qs = np.random.random((5000, 3))
    test_atoms = np.random.random((1000, 3))
    test_Tu = np.random.random((500, 3))
    rand_rot_mat = sp.spatial.transform.Rotation.random(1)
    rotat_mat = rand_rot_mat.as_matrix()[0]

    mol = older.molecular_transform_no_loop_array_3d(test_qs, test_atoms, rotat_mat)
    mol_chunk = chunk_transform(test_qs, test_atoms, rotat_mat, 70)
    assert np.allclose(mol, mol_chunk)
    print("Molecular transform pass!")

    lat = older.lattice_transform_no_loop_array_3d(test_qs, test_Tu, rotat_mat)
    lat_chunk = chunk_transform(test_qs, test_Tu, rotat_mat, True)
    assert np.allclose(lat, lat_chunk)
    print("Lattice transform pass!")
