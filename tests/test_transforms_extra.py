import numpy as np
from toysim.legacy import older


def test_molecular_chunking_equivalence():
    """Compare two legacy molecular transform implementations (chunked vs array).
    Both functions live in `legacy/older.py` and should produce identical outputs.
    """
    np.random.seed(0)
    Qs = np.random.random((100, 3))
    Atoms = np.random.random((40, 3))
    rand_rot = np.eye(3)
    chunk_size = 10

    a_chunks = older.molecular_transform_chunks(Qs, Atoms, rand_rot, chunk_size)
    a_array = older.molecular_transform_no_loop_array_3d(Qs, Atoms, rand_rot)

    assert np.allclose(a_chunks, a_array)


def test_lattice_chunking_equivalence_manual():
    """Verify lattice sum computed in one shot equals manual chunked accumulation."""
    np.random.seed(1)
    Qs = np.random.random((80, 3))
    Tu = np.random.random((20, 3))
    rotation_m = np.eye(3)
    chunk_size = 5

    # reference (one-shot) result
    lat_ref = older.lattice_transform_no_loop_array_3d(Qs, Tu, rotation_m)

    # manual chunk accumulation: split Tu into chunks and sum contributions
    num_chunks = len(Tu) // chunk_size
    if num_chunks == 0:
        num_chunks = 1
    chunks = np.array_split(Tu, num_chunks)
    a_acc = np.zeros(len(Qs))
    b_acc = np.zeros(len(Qs))
    for chunk in chunks:
        rotated_u = np.dot(rotation_m, chunk.T)
        phase = np.dot(Qs, rotated_u)
        a_acc += np.sum(np.cos(phase), axis=1)
        b_acc += np.sum(np.sin(phase), axis=1)
    lat_chunked = a_acc + 1j * b_acc

    assert np.allclose(lat_ref, lat_chunked)


def test_q_vector_creators_shapes():
    q2 = older.create_q_vectors(1, step_size=1)
    assert q2.shape == (9, 2)

    q3 = older.create_q_vectors_3d(1, step_size=1)
    assert q3.shape == (9, 3)
    # z column should equal sqrt(x^2+y^2)
    assert np.allclose(q3[:, 2], np.sqrt(q3[:, 0] ** 2 + q3[:, 1] ** 2))
