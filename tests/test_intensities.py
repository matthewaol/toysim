from toysim.simulation import crystal_intensities_gpu, crystal_intensities
from toysim.simulation import gpu_transform
import numpy as np
from toysim import lattice
from toysim import models, detectors
from dxtbx.model import DetectorFactory, BeamFactory
from toysim import gpu
from toysim.simulation import chunk_transform


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


class TestIntensities:

    def setup(self):
        beam = BeamFactory.from_dict(models.beam)
        det = DetectorFactory.from_dict(models.pilatus)
        self.qvecs, _ = detectors.qxyz_from_det(det, beam)
        self.qvecs = self.qvecs[0]
        self.qvecs = self.qvecs[::100]  # simulate every 100th q
        assert len(self.qvecs.shape) == 2
        self.Tu = lattice.create_Tu_vectors_3d(2, 100)
        assert len(self.Tu.shape) == 2
        box_size = 30
        self.sample_atoms = np.random.random((10,3))*box_size
        self.rotate_mat = np.eye(3)

        self.K_sol = 0.8
        self.B_sol = 100

    def test_solvent_term(self):
        self.setup()  # TODO: is this necessary ??

        I = crystal_intensities(self.qvecs, self.sample_atoms, self.Tu, self.rotate_mat,
                            lat_chunk_size=2, molec_chunk_size=3,
                            K_sol=self.K_sol, B_sol=self.B_sol, solvent_term=True)

        I2 = crystal_intensities(self.qvecs, self.sample_atoms, self.Tu, self.rotate_mat,
                                lat_chunk_size=2, molec_chunk_size=3,
                                K_sol=self.K_sol, B_sol=self.B_sol, solvent_term=False)

        assert not np.allclose(I, I2)
        qmags = np.linalg.norm(self.qvecs)
        idx_of_min_q = np.argmin(qmags)

        I_at_min_q = I[idx_of_min_q]
        I2_at_min_q = I2[idx_of_min_q]

        assert I2_at_min_q > I_at_min_q

    def test_gpu_phase_sum(self):
        self.setup()

        c,q = gpu.get_context_queue()
        gpu_helper = gpu.GPUHelper(self.qvecs, c, q)
        amp_real, amp_imag = gpu_helper.phase_sum(self.sample_atoms)
        assert not np.allclose(amp_real, 0)
        assert not np.allclose(amp_imag, 0)

        # now test CPU code
        amp2 = chunk_transform(self.qvecs, self.sample_atoms, self.rotate_mat, chunk_size=3, f_j_is1=True)
        assert not np.allclose(amp2.real, 0)
        assert not np.allclose(amp2.imag, 0)

        assert np.allclose(amp_real, amp2.real)
        assert np.allclose(amp_imag, amp2.imag)

        amp3 = gpu_transform(gpu_helper, self.sample_atoms, self.rotate_mat, f_j_is1=True)

        # just calling kernel again causes the above test to fail ????!!!!
        assert np.allclose(amp_real, amp2.real)
        assert np.allclose(amp_imag, amp2.imag)

        assert np.allclose(amp2.real, amp3.real)
        assert np.allclose(amp2.imag, amp3.imag)

        latt_cpu = chunk_transform(self.qvecs, self.Tu, self.rotate_mat, chunk_size=3, f_j_is1=True)
        latt_gpu = gpu_transform(gpu_helper, self.Tu, self.rotate_mat, f_j_is1=True)

        assert np.allclose(latt_cpu.real, latt_gpu.real)
        assert np.allclose(latt_cpu.imag, latt_gpu.imag)
        print('ok')

    def test_crystal_intensities_gpu(self):
        self.setup()  # TODO: is this necessary ??

        c,q = gpu.get_context_queue()
        gpu_helper = gpu.GPUHelper(self.qvecs, c, q)

        I = crystal_intensities(self.qvecs, self.sample_atoms, self.Tu, self.rotate_mat,
                                lat_chunk_size=2, molec_chunk_size=3,
                                K_sol=self.K_sol, B_sol=self.B_sol, solvent_term=False)
        assert not np.allclose(I, 0)

        I_gpu = crystal_intensities_gpu(gpu_helper, self.qvecs, self.sample_atoms, self.Tu, self.rotate_mat,
                                K_sol=self.K_sol, B_sol=self.B_sol, solvent_term=False)

        assert not np.allclose(I_gpu, 0)

        assert np.allclose(I, I_gpu)


if __name__=="__main__":
    test_crystal_intensities_length()
    TI = TestIntensities()
    TI.test_gpu_phase_sum()
