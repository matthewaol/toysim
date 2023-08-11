#!/usr/bin/env python
# coding: utf-8

import pylab as plt 
import scipy as sp
import numexpr as ne

import time 
import bgnoise
import models
import detectors
from dxtbx.model.beam import BeamFactory
from dxtbx.model.detector import DetectorFactory
from toysim import molecule, lattice

import numpy as np


def gpu_transform(gpu_helper, vectors, rotation_m, f_j_is1=True):
    """

    Parameters
    ----------
    gpu_helper
    vectors
    rotation_m
    f_j_is1

    Returns
    -------

    """
    if not f_j_is1:
        # TODO: store f_j as class attribute of gpu_helper
        num_q = int(len(gpu_helper.q_vecs) / 3)
        Qs = np.reshape(gpu_helper.q_vecs, (num_q, 3))
        Qs_mag = np.linalg.norm(Qs, axis=1)
        exp_arg = Qs_mag ** 2 / 4 * -10.7
        f_j = 7 * np.exp(exp_arg)
    else:
        f_j = 1

    # rotate the atom
    rotated_u = np.dot(rotation_m, vectors.T)
    # run the GPU kernel
    a2, b2 = gpu_helper.phase_sum(rotated_u)

    i_real, i_imag = a2 * f_j, b2 * f_j
    return i_real + i_imag * 1j


def chunk_transform(Qs, realsp_vectors, rotation_m, chunk_size, f_j_is1=True):
    """

    Parameters
    ----------
    Qs: reciprocal space vectors
    realsp_vectors: either atomic coordinates or lattice coordinates
    rotation_m
    chunk_size
    f_j_is1

    Returns
    -------

    """
    real_part = np.zeros(len(Qs))
    imag_part = np.zeros(len(Qs))
    
    if not f_j_is1:
        # TODO: check scaling with 2PI!
        Qs_mag = np.linalg.norm(Qs, axis=1)
        stol = Qs_mag / 2
        exp_arg = stol**2 * -10.7
        f_j = 7 * np.exp(exp_arg)
    else:
        f_j = 1

    num_chunks = len(realsp_vectors) / chunk_size

    print("Chunking up")
    chunked_vectors = (np.array_split(realsp_vectors, num_chunks))
    
    for i_chunk, chunk in enumerate(chunked_vectors):
        print("Computing chunk #", i_chunk+1, "out of", len(chunked_vectors))

        rotated_u = np.dot(rotation_m, chunk.T)
        chunk_phase = np.dot(2*np.pi*Qs, rotated_u)
        cos_phase = ne.evaluate("cos(chunk_phase)")
        sin_phase = ne.evaluate("sin(chunk_phase)")
        real_part += np.sum(cos_phase, axis=1)
        imag_part += np.sum(sin_phase, axis=1)

    real_part *= f_j
    imag_part *= f_j
    return real_part + imag_part*1j


def crystal_intensities(Qs, Atoms, Tu, rotation_m,
                        molec_chunk_size,
                        lat_chunk_size, K_sol=.85, B_sol=200,
                        solvent_term=True):
    """
    Parameters
    ----------
    Qs
    Atoms
    Tu
    rotation_m
    molec_chunk_size
    lat_chunk_size
    K_sol
    B_sol
    solvent_term

    Returns
    -------
    numpy array of intensities (of length equal to Qs)
    """
    # simulation variables

    print("Computing intensities")
    print("Computing Molecular transform")
    a_molecular = chunk_transform(Qs, Atoms, rotation_m, molec_chunk_size, f_j_is1=True)

    # Solvent calculation
    if solvent_term:
        q_mags = np.linalg.norm(Qs,axis=1)
        F = np.sqrt(a_molecular.real**2 + a_molecular.imag**2)
        exp_term = np.exp(-B_sol * q_mags**2 / 4)
        theta = np.arctan2(a_molecular.imag, a_molecular.real)
        F = (1-K_sol*exp_term) * F

        a_molecular = F*(np.cos(theta) + 1j* np.sin(theta))

    print("Computing Lattice transform")
    a_lattice = chunk_transform(Qs, Tu, rotation_m, lat_chunk_size, f_j_is1=True)

    a_total = a_molecular * a_lattice
    
    print("Finished intensities")
    return a_total.real**2 + a_total.imag**2


if __name__ == '__main__':
    start = time.time()
    # PARAMETERS TO VARY IMAGE
    K_sol = 0.85  # solvent scale
    B_sol = 200  # solvent decay
    Bfactor_A = 4  # overall crystal B-factor
    Rotation_seed = 0  # soecfies a randomly sampled rotation matrix
    PDB_file = "4bs7.pdb"  # moleculr coordinates
    background_file = "randomstols/water_014.stol"  # sets the background model
    random_rad = 300  # random radius on detector to scale background at
    num_cells = 7  # number of unit cells along each dimension
    n_atoms = None  # set to an int to only simulate that many atoms
    # end PARAMETERS TO VARY IMAGE

    print("Starting 3D simulation")

    print("Bringing in Qs from detector")
    beam = BeamFactory.from_dict(models.beam) 
    detector = DetectorFactory.from_dict(models.pilatus)
    # TO DO: Shift detector z
    # Add Mosaic texture

    qvecs, img_sh = detectors.qxyz_from_det(detector, beam)
    qvecs = qvecs[0]  # specifies q-vectors

    rand_rot_mat = sp.spatial.transform.Rotation.random(1, random_state=Rotation_seed)
    rotate_mat = rand_rot_mat.as_matrix()[0]

    sample_atoms = molecule.get_coords_pdb(PDB_file, "temp", get_only_xy=False, apply_sym_ops=True)
    if n_atoms is not None:
        sample_atoms = sample_atoms[:n_atoms]

    M = molecule.get_M(PDB_file)
    Mi = np.linalg.inv(M)
    a,b,c = Mi.T
    asize = np.linalg.norm(a)
    bsize = np.linalg.norm(b)
    csize = np.linalg.norm(c)
    print("Length of a,b,c= %.1f, %.1f, %.1f" % (asize, bsize, csize))
    Tu = lattice.create_Tu_vectors_3d_basis(num_cells, a, b, c)
    #Tu = lattice.create_Tu_vectors_3d(num_cells, cell_size=200)

    I = crystal_intensities(qvecs, sample_atoms, Tu, rotate_mat,
                            lat_chunk_size=20, molec_chunk_size=20,
                            K_sol=K_sol, B_sol=B_sol, solvent_term=True)

    # add a B-factor decay to the intensities
    I = bgnoise.add_background_exp_no_loop(I, qvecs, a=Bfactor_A)

    # get background
    q_mags = np.linalg.norm(qvecs, axis=1)
    background = bgnoise.add_background_file(background_file, q_mags)

    # shape I and background into a 2D image
    I = np.reshape(I, img_sh)
    background = np.reshape(background, img_sh)

    # scale the background at a random rad
    total_I = bgnoise.scale_background_list_r(I, background, radius=random_rad)

    end = time.time()
    print("Program took %.1f sec or %.2f minutes" % (end-start, (end-start)/60))
    print("# of pixels:", len(qvecs))
    print("# of unit cells", len(Tu))
    m = I.mean()
    s = I.std()
    vmin=m-s
    vmax=m+3*s
    plt.imshow(I, vmax=vmax, vmin=vmin)
    plt.show()
    from IPython import embed;embed()