from toysim.simulation import crystal_intensities
import numpy as np
from toysim import lattice


def test_crystal_intensities_length():

    qvecs = np.random.random((100,3))
    sample_atoms = np.random.random((10,3))
    Tu = np.random.random((30,3))
    rotate_mat = np.eye(3)
    K_sol = 0.8
    B_sol = 100
    I = crystal_intensities(qvecs, sample_atoms, Tu, rotate_mat,
                            lat_chunk_size=2, molec_chunk_size=3,
                            K_sol=K_sol, B_sol=B_sol, solvent_term=True)
    assert len(I) == len(qvecs)


from toysim import models, detectors
from dxtbx.model import DetectorFactory, BeamFactory
def test_solvent_term():

    beam = BeamFactory.from_dict(models.beam)
    det = DetectorFactory.from_dict(models.pilatus)
    qvecs, _ = detectors.qxyz_from_det(det, beam)
    qvecs = qvecs[0]
    qvecs = qvecs[::100]  # simulate every 100th q
    assert len(qvecs.shape)==2
    Tu = lattice.create_Tu_vectors_3d(2, 100)
    assert len(Tu.shape)==2
    box_size = 30
    sample_atoms = np.random.random((10,3))*box_size
    rotate_mat = np.eye(3)

    K_sol=0.8
    B_sol=100
    I = crystal_intensities(qvecs, sample_atoms, Tu, rotate_mat,
                            lat_chunk_size=2, molec_chunk_size=3,
                            K_sol=K_sol, B_sol=B_sol, solvent_term=True)

    I_noSolv = crystal_intensities(qvecs, sample_atoms, Tu, rotate_mat,
                            lat_chunk_size=2, molec_chunk_size=3,
                            K_sol=K_sol, B_sol=B_sol, solvent_term=False)
    assert not np.allclose(I, I_noSolv)
    qmags = np.linalg.norm(qvecs)
    idx_of_min_q = np.argmin(qmags)

    I_at_min_q = I[idx_of_min_q]
    I2_at_min_q = I_noSolv[idx_of_min_q]

    assert I2_at_min_q > I_at_min_q


if __name__=="__main__":
    test_crystal_intensities_length()
    test_solvent_term()
