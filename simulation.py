#!/usr/bin/env python
# coding: utf-8

import pylab as plt 
import numpy as np
import scipy as sp
import numexpr as ne

import time 
from Bio.PDB.PDBParser import PDBParser
from iotbx import pdb

from cctbx import sgtbx
import bgnoise
import models
import detectors
from dxtbx.model.beam import BeamFactory
from dxtbx.model.detector import DetectorFactory 

def find_sym_info(file_name): # returns space group info 
    P = pdb.input(file_name)
    symbol = P.crystal_symmetry().space_group_info().type().lookup_symbol()
    return symbol

def get_sym_operators(space_group_info): # gets arrays of sym operators
    sg = sgtbx.space_group_info(space_group_info)
    gr = sg.group()
    Ops = gr.all_ops()
    trans = [np.reshape(O.t().as_double(), (1,3)) for O in Ops] 
    rots = [np.reshape(O.r().as_double(), (3,3)) for O in Ops]

    return np.array(trans), np.array(rots)

def get_coords_pdb(file_name, structure_id,get_only_xy=True):
    '''
    Gets atom coordinates from a .pdb file
    Parameters: 
        file_name: string path to file
        structure_id: string id of structure
        get_only_xy: Assign false if xyz coordinates are needed 
    Returns:
        numpy array of atom coordinates 
    '''
    print("Getting atom coordinates from molecule")
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(structure_id,file_name)

    space_group = find_sym_info(file_name)
    sym_ops = get_sym_operators(space_group)

    atoms = structure.get_atoms()

    list_of_coords = [atom.get_coord() for atom in atoms]
    
    print("Applying symmetry operators")
    translated = []
    
    #still implementing
    for trans in sym_ops[0]: 
        for rot in sym_ops[1]:
            translated.append(np.dot((list_of_coords + trans), rot))

    if get_only_xy is True: 
        list_of_coords = np.array(np.delete(list_of_coords, 2, axis=1))

    from IPython import embed;embed()
    return np.array(translated)

def show_image(square_I_list): 
    '''
    Displays a square 2D array of values as an a image and uses its mean and std to determine vmax and vmin 
    Parameters:
        square_I_list: 2D reshaped array of image values
    '''
    print("Showing Image")
    mean = square_I_list.mean()
    std = square_I_list.std()
    
    vmin = mean - std
    vmax = mean + std 
    
    photo = plt.imshow(square_I_list, vmax = vmax, vmin = vmin)
    plt.show()

# Q-vector functions
def create_q_vectors(image_size, step_size=1):
    '''
    Creates 2d array of Q-vectors ranging from [-image_size,-image_size] to [image_size,image_size]
    Parameters: 
        image_size: int specifiying the size of q-vectors
        step_size: float specifying the step size between q-vectors (default is 1)
    Returns:
        2D numpy array of Q-vectors
    '''
    q_vectors = [] 
    
    for x in np.arange(-image_size, image_size+.01, step_size): 
        for y in np.arange(-image_size, image_size+.01, step_size):  
            q_vectors.append([x, y])
    return np.array(q_vectors) 


def create_q_vectors_3d(image_size, step_size): 
    q_vectors = [] 
    
    for x in np.arange(-image_size, image_size+.01, step_size): 
        for y in np.arange(-image_size, image_size+.01, step_size):  
                q_vectors.append([x, y, np.sqrt(x**2 + y**2)])
    return np.array(q_vectors)
 
# Lattice functions
def create_Tu_vectors(num_cells, cell_size=1): 
    '''
    Creates lattice vectors with unit cells forming a square
        Parameters: 
            num_cells: int specifying the number of unit cells in lattice
            cell_size: float specifying the size of each cell (cell_size by cell_size) (default=1)
        Returns:
            2D numpy array of lattice vectors
    '''
    Tu = []
    
    for x in range(num_cells):
        for y in range(num_cells):
            Tu.append([x * cell_size, y * cell_size])
    return np.array(Tu)

def create_Tu_vectors_3d(num_cells, cell_size=1): 
    '''
    Creates lattice vectors with unit cells forming a cube
        Parameters: 
            num_cells: int specifying the number of unit cells in lattice
            cell_size: float specifying the size of each cell (cell_size by cell_size by cell_size) (default=1)
        Returns:
            2D numpy array of lattice vectors
    '''
    Tu = []
    
    for x in range(num_cells): # Creates Tu vectors ranging from [0,0] to [Tu_size,Tu_size]
        for y in range(num_cells):
                for z in range(num_cells):
                    Tu.append([x*cell_size,y*cell_size,z*cell_size])
    return np.array(Tu)

def create_Tu_vectors_3d_basis(num_cells, a_vec, b_vec, c_vec): 
    ...
    # Tu_vec = x* a_vec + y* b_vec + z*c_vec
    # a_vec are 1d numpy arrays
    Tu = []
    for x in range(num_cells): # Creates Tu vectors ranging from [0,0] to [Tu_size,Tu_size]
        for y in range(num_cells):
            for z in range(num_cells):
                Tu.append([x*a_vec,y*b_vec,z*c_vec])
    return np.array(Tu)

def create_Tu_vectors_3d_tetra(num_cells,a,c): 
    Tu = []

    for x in range(num_cells):
        for y in range(num_cells):
            for z in range(num_cells):
                Tu.append([x*a, y*a, z*c])
    return np.array(Tu)

def create_Tu_vectors_3d_ortho(num_cells,a,b,c):
    for x in range(num_cells):
        for y in range(num_cells):
            for z in range(num_cells):
                Tu.append([x*a, y*b, z*c])
    return np.array(Tu)

def determine_cell_size(atoms): 
    '''
    Determines an appropriate cell size for lattice based on the maximum distance value from array of atoms
        Parameters: 
            atoms: Array of atom coordinates 
        Returns: float value of the maximum distance found in array of atom coordinates
    '''
    return sp.spatial.distance.pdist(atoms).max() * 5

# Molecular Transform functions
def molecular_transform(Q, Atoms,f_j,theta): # takes in a single Q vector and a list of atoms, outputs a single A value
    a = b = 0
    
    for atom in Atoms:
        phase = plt.dot(Q,rotation_matrix(atom,theta)) 
        a+= plt.cos(phase)  
        b+= plt.sin(phase) 
    
    i_real, i_imag = a*f_j, b*f_j
    return i_real + i_imag*1j

def molecular_transform_no_loop(Q,Atoms,f_j,theta): # takes in a single Q vector and a list of atoms, outputs a single A value
    a = b = 0
    rotation_m = [[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]
    
    rotated_u = np.dot(rotation_m,Atoms.T)
    phase = np.dot(Q, rotated_u)
    a = sum(np.cos(phase))
    b = sum(np.sin(phase))
    i_real, i_imag = a*f_j, b*f_j
    return i_real + i_imag*1j

def molecular_transform_no_loop_array(Qs,Atoms,f_j,theta): # takes in an array of Q instead of a single Q
    a = b = 0
    rotation_m = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    
    rotated_u = np.dot(rotation_m,Atoms.T)
    phase = np.dot(Qs, rotated_u)
    a = np.sum(np.cos(phase), axis = 1) # indicating axis = 1, to sum over the columns instead of rows
    b = np.sum(np.sin(phase), axis = 1)
    i_real, i_imag = a*f_j, b*f_j
    return i_real + i_imag*1j

def molecular_transform_no_loop_array_3d(Qs, Atoms,rotation_m): 
    '''
    Computes molecular transform on array of Q-vectors and atoms
        Parameters: 
            Qs: Array of 3D Q-vectors
            Atoms: Array of 3D atoms coordinates 
            rotation_m: 3D rotation matrix
        Returns:
            Array of molecular transform values in complex num form
    '''
    a = b = 0 
    
    Qs_mag = np.linalg.norm(Qs,axis=1)
    exp_arg = Qs_mag**2 / 4 * -10.7
    f_j = 7 * np.exp(exp_arg)

    rotated_u = np.dot(rotation_m, Atoms.T)
    phase = np.dot(Qs, rotated_u)
    
    cos_phase = ne.evaluate("cos(phase)")
    sin_phase = ne.evaluate("sin(phase)")

    a = np.sum(cos_phase, axis = 1)
    b = np.sum(sin_phase, axis = 1) 

    i_real, i_imag = a*f_j, b*f_j

    return i_real + i_imag*1j

#@profile
def molecular_transform_chunks(Qs, Atoms, rotation_m, chunk_size): 
    a2 = np.zeros(len(Qs))
    b2 = np.zeros(len(Qs))
    
    Qs_mag = np.linalg.norm(Qs,axis=1)
    exp_arg = Qs_mag**2 / 4 * -10.7
    f_j = 7 * np.exp(exp_arg)

    num_chunks = len(Atoms) / chunk_size

    print("Chunking up atoms")
    chunked_atoms = (np.array_split(Atoms,num_chunks))
    print("Computing molecular transform for each chunk")
    
    i=0
    for chunk in chunked_atoms:
        i+=1
        print("Computing chunk #",i)

        rotated_u = np.dot(rotation_m, chunk.T) 
        chunk_phase = np.dot(Qs, rotated_u)

        a2 += np.sum(ne.evaluate("cos(chunk_phase)"), axis=1) 
        b2 += np.sum(ne.evaluate("sin(chunk_phase)"), axis=1)

    i_real, i_imag = a2*f_j, b2*f_j
    return i_real + i_imag*1j

def test_molecular_transform():
    '''
    Testing the molecular transform withh the chunking version 
    '''
    test_qs = np.random.random((5000,3))#create_q_vectors_3d(20,.1)
    test_atoms = np.random.random((1000,3))#np.array([[3,4,1],[3,1,6],[2,1,5],[3,2,1]])
    rand_rot_mat = sp.spatial.transform.Rotation.random(1,random_state=0)
    rotat_mat = rand_rot_mat.as_matrix()[0]
    molecular = molecular_transform_no_loop_array_3d(test_qs,test_atoms, rotat_mat)
    molecular_chunks = molecular_transform_chunks(test_qs,test_atoms,rotat_mat, 100)
    assert np.allclose(molecular,molecular_chunks)
    print("Molecular transform passed!")

# Lattice transform functions
def lattice_transform(Q, Tu,theta): # takes in a single Q vector and a list of Tu vectors, outputs a single A value
    a = b = 0       
    
    for u in Tu:
        # u is the vector representing the distance between the origin and the corner of a lattice unit
        phase = plt.dot(Q,rotation_matrix(u,theta))
        a+= plt.cos(phase)
        b+= plt.sin(phase)
    i_real, i_imag = a,b
    return i_real + i_imag*1j 

def lattice_transform_no_loop(Q, Tu, theta): # takes in single Q vector and list of Tu vectors, outputs a single A value
    rotation_m = [[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]
    
    rotated_u = np.dot(rotation_m, Tu.T) # Applying rotation to array of Tu vectors
    phase = np.dot(Q, rotated_u) # applying dot product to a Q vector and rotated Tu vectors
    a = sum(np.cos(phase)) # cosine of phase
    b = sum(np.sin(phase)) # sin of phase
    i_real, i_imag = a, b
    return i_real + i_imag * 1j 

def lattice_transform_no_loop_array(Qs, Tu, theta): 
    '''
    Performs lattice transform with Q-vectors and Lattice vectors
        Parameters: 
            Qs: 2D array of 2D q-vectors
            Tu: 2D array of Tu (lattice) vectors
            theta: angle in radians to rotate transform
        Returns: 1D array of lattice transform values in complex number form
    '''
    rotation_m = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    rotated_u = np.dot(rotation_m,Tu.T) 
    phase = np.dot(Qs, rotated_u)  
    a = np.sum(np.cos(phase), axis = 1) 
    b = np.sum(np.sin(phase), axis = 1)

    i_real, i_imag = a, b
    return i_real + i_imag * 1j

def lattice_transform_no_loop_array_3d(Qs, Tu, rotation_m): # 3d Qs & Tu, takes in rotation matrix    
    a = b = 0 
    
    rotated_u = np.dot(rotation_m,Tu.T) 
    phase = np.dot(Qs, rotated_u)  
        
    cos_phase = ne.evaluate("cos(phase)")
    sin_phase = ne.evaluate("sin(phase)")

    a = np.sum(cos_phase, axis = 1) # indicating axis = 1, to sum over the columns instead of rows
    b = np.sum(sin_phase, axis = 1)

    i_real, i_imag = a, b
    return i_real + i_imag * 1j

def chunk_transform(Qs,vectors,rotation_m,chunk_size,f_j_is1=True):
    a2 = np.zeros(len(Qs))
    b2 = np.zeros(len(Qs))
    
    if f_j_is1 is not True:
        Qs_mag = np.linalg.norm(Qs,axis=1)
        exp_arg = Qs_mag**2 / 4 * -10.7
        f_j = 7 * np.exp(exp_arg)
    else:
        f_j=1

    num_chunks = len(vectors) / chunk_size

    print("Chunking up")
    chunked_vectors = (np.array_split(vectors,num_chunks))
    
    i=0
    for chunk in chunked_vectors:
        i+=1
        print("Computing chunk #",i)

        rotated_u = np.dot(rotation_m, chunk.T) 
        chunk_phase = np.dot(Qs, rotated_u)

        a2 += np.sum(ne.evaluate("cos(chunk_phase)"), axis=1) 
        b2 += np.sum(ne.evaluate("sin(chunk_phase)"), axis=1)

    i_real, i_imag = a2*f_j, b2*f_j
    return i_real + i_imag*1j

def test_chunk_transform():
    test_qs = np.random.random((5000,3))
    test_atoms = np.random.random((1000,3))
    test_Tu = np.random.random((500,3))
    rand_rot_mat = sp.spatial.transform.Rotation.random(1)
    rotat_mat = rand_rot_mat.as_matrix()[0]

    mol = molecular_transform_no_loop_array_3d(test_qs,test_atoms,rotat_mat)
    mol_chunk = chunk_transform(test_qs, test_atoms,rotat_mat,70)
    assert np.allclose(mol,mol_chunk)
    print("Molecular transform pass!")
    
    lat = lattice_transform_no_loop_array_3d(test_qs,test_Tu,rotat_mat)
    lat_chunk = chunk_transform(test_qs, test_Tu, rotat_mat,True)
    assert np.allclose(lat,lat_chunk)
    print("Lattice transform pass!")
    
# Complete transform functions
def get_I_values(Qs, Atoms, Tu, f_j, theta): # calculating the lattice and molecular transforms and returning intensities
    
    print("Computing intensities")
    
    I_list = []
    for Q in Qs: # for each Q value do these steps! 
        a_molecular = molecular_transform_no_loop(Q,Atoms,f_j,theta) # apply molecular transform! 
        a_lattice = lattice_transform_no_loop(Q,Tu,theta) # apply lattice transform!
        a_total = a_molecular * a_lattice # multiply them together
        final_I = a_total.real**2 + a_total.imag**2 # add the two complex numbers
        I_list.append(final_I) # append that to list
    
    print("Finished intensities")
    
    return np.array(I_list)

def get_I_values_no_loop(Qs, Atoms, Tu, f_j, theta): 
    
    print("Computing intensities")
    
    a_molecular = molecular_transform_no_loop_array(Qs, Atoms, f_j, theta)
    a_lattice = lattice_transform_no_loop_array(Qs, Tu, theta)
    a_total = a_molecular * a_lattice
    
    print("Finished intensities")
    return a_total.real**2 + a_total.imag**2

def get_I_values_no_loop_3d(Qs, Atoms, Tu, rotation_m): 
    
    print("Computing intensities")
    
    a_molecular = molecular_transform_no_loop_array_3d(Qs, Atoms, rotation_m)
    a_lattice = lattice_transform_no_loop_array_3d(Qs, Tu, rotation_m)
    a_total = a_molecular * a_lattice
    
    print("Finished intensities")
    return a_total.real**2 + a_total.imag**2

def get_I_values_no_loop_3d_chunks(Qs, Atoms, Tu, rotation_m,molec_chunk_size,lat_chunk_size): 
    
    print("Computing intensities")

    print("Computing Molecular transform")
    a_molecular = chunk_transform(Qs, Atoms, rotation_m,molec_chunk_size)

    # Solvent calculation
    q_mags = np.linalg.norm(Qs,axis=1) 
    F = np.sqrt(a_molecular.real**2 + a_molecular.imag**2)
    K_sol = 0.85
    B_sol = 200
    exp_term = np.exp(-B_sol * q_mags**2 / 4) 

    theta = np.arctan(a_molecular.imag / a_molecular.real)

    F = (1-K_sol*exp_term) * F

    a_molecular.real = F * np.cos(theta) 
    a_molecular.imag = F * np.sin(theta) 



    print("Computing Lattice transform")
    a_lattice = chunk_transform(Qs, Tu, rotation_m,lat_chunk_size,True)
    a_total = a_molecular * a_lattice
    
    print("Finished intensities")
    return a_total.real**2 + a_total.imag**2

def count_spots(square_I_list, threshold_factor=None):
    if threshold_factor is not None: 
        threshold = square_I_list > threshold_factor 
    else:
        threshold = square_I_list 
        
    labels, num_of_labels = sp.ndimage.label(threshold) # Label each index of intensity array based on our defined threshold
    peaks = sp.ndimage.find_objects(labels) # gets tuples corresponding to the location of each peak
    print("Number of spots: " + str(num_of_labels)) 
    return num_of_labels, peaks 

def produce_image(pdb_file_name, Qs, num_cells,cell_size, a, degrees): 
    '''
    Produces a simulated diffraction image from 2D Q-vectors and a pdb file
        Parameters: 
            pdb_file_name: string path to pdb file
            Qs: 2D array of Q-vectors
            num_cells: int specifying the number of cells for lattice
            cell_size: float specifiying the cell size for lattice
            a: float value constant for controlling the background 
            degrees: float value for rotating the image
        Returns:
            List of intensity values reshaped into a square
    '''
    f_j = 1
    Tu = create_Tu_vectors(num_cells,cell_size)
    
    theta = degrees * np.pi / 180
    molecule = get_coords_pdb(pdb_file_name, "temporary_id")
    
    I_list = get_I_values_no_loop(Qs, molecule, Tu, f_j,theta)
    I_list_background = bgnoise.add_background_exp_no_loop(I_list,Qs,a)
    
    I_list_size = int(np.sqrt(len(I_list)))
    
    square_I_list = plt.reshape(I_list_background, (I_list_size,I_list_size) )
    
    return square_I_list

if __name__ == '__main__':
    #3D sim
    start = time.time()

    print("Starting 3D simulation")

    print("Bringing in Qs from detector")
    beam = BeamFactory.from_dict(models.beam) 
    detector = DetectorFactory.from_dict(models.pilatus)
    qvecs, img_sh = detectors.qxyz_from_det(detector, beam)
    qvecs = qvecs[0]

    rand_rot_mat = sp.spatial.transform.Rotation.random(1,random_state=0)
    rotat_mat = rand_rot_mat.as_matrix()[0]

    rotat_mat=1
    sample_atoms = get_coords_pdb("4bs7.pdb", "temp", False)[:200]

    vecs = create_q_vectors_3d(20,.2)
    Tu = create_Tu_vectors_3d(5, determine_cell_size(sample_atoms))


    I_list = get_I_values_no_loop_3d_chunks(qvecs, sample_atoms, Tu, rotat_mat, 35, 50)
    #B_I_list = bgnoise.add_background_exp_no_loop(I_list,qvecs,a=4)

    background = bgnoise.add_background_file("randomstols/water_014.stol",np.linalg.norm(qvecs,axis=1))
    
    shaped_I_list = np.reshape(I_list, (img_sh))
    shaped_background = np.reshape(background, (img_sh))
    
    random_rad = np.random.randint(150,600)

    shaped_background_I_list = bgnoise.scale_background_list_r(shaped_I_list, shaped_background,300)

    end = time.time()
    print("Program took", str(end-start), "secs or", ((end-start)/60 ), "minutes") 

    plt.imshow(shaped_background_I_list, vmax=1e10)
    plt.show()
    from IPython import embed;embed()

    #2D sim
    # print("Starting 2D simulation")
    # image_size = 20 # Parameter - Image Size (for now = 40) 
    # image_resolution = .2 # Parameter - Step size of image (for now = .5) / went from .5 to .2 to reduce distortion from rotation

    # Qs = create_q_vectors(image_size, image_resolution)

    # Qs_size = int(np.sqrt(len(Qs))) # this gives us the size of the image! eg. 3 -> 3x3 image
    
    # triangle = plt.array([[1,1.5],[1.5,0],[0,1]]) # Parameter - Atoms
    # molecule = get_coords_pdb("4bs7.pdb", "4bs7")

    # f_j = 1

    # num_cells, cell_size= 5, determine_cell_size(molecule) # Parameter - Tu Vectors size 
    # Tu = create_Tu_vectors(num_cells,cell_size)

    # degrees = 0
    # theta = degrees * np.pi / 180 # Parameter - theta degrees (rad) to rotate vectors

    # a = .003 # Parameter - value for the background, decent results are between .001 to .009

    # I_list = get_I_values_no_loop(Qs, molecule, Tu, f_j,theta) # Computing intensity values

    # I_list = bgnoise.add_background_waterfile("gauss_bg_for_2Dcase.txt", I_list,np.linalg.norm(Qs,axis=1),100)
    # #I_list = bgnoise.add_background_water_offset(I_list,np.linalg.norm(Qs,axis=1))
                                                            
    # square_I_list = plt.reshape(I_list, (Qs_size,Qs_size)) # reshaping list into a square 
    
    # #spot_count = count_spots(I_list, square_I_list)

    # plt.imshow(square_I_list,vmax=1e6)
    # plt.show()

    # Background movie! going from a = 0 to 0.06 in steps of .001
    # for i in np.arange(0,.06,.001):
    #     a = i 
    #     I_background_list = add_background_exp(I_list,Qs,a)
    #     square_I_list = plt.reshape(I_background_list, (Qs_size,Qs_size))
    #     show_image(square_I_list)

    # Rotation movie!  going from 0 to 90 degrees in steps of 10 degrees
    # print("starting rotation movie")
    # for i in np.arange(10): 
    #     degrees = 10 * i * np.pi / 180
    #     img = get_I_values(Qs,molecule,Tu,1,degrees)
    #     square_I_list = plt.reshape(img, (Qs_size,Qs_size)) 
    #     plt.imshow(square_I_list, vmax = 1e6)
    #     plt.draw
    #     plt.pause(.5)

    #Trying to add noise : have gaussian, poisson, saltpepper so far
    #mu, sigma = 0, .1 # where mu = mean, and sigma = standard deviation (sigma !< 0)
    #gaussian_I_list = add_gaussian_noise(I_list,mu,sigma)

    #lam = 1 # where lam = average occurences of event within given timeframe
    #poisson_I_list = add_poisson_noise(I_list,lam) 

    #noise_intensity = 6000 # = num of iterations of a random pixel getting replaced
    #saltpepper_I_list = add_saltpepper_noise(I_list, noise_intensity) # for some reason this keeps applying to the og list

    # print("Size of image is " + str(Qs_size) + " by " + str(Qs_size))
    # print("Length of Qs is: " + str(Qs_len))
    # print("Gaussian Noise: " + "mu = " + str(mu) + " & sigma = " + str(sigma))
    # print("Poisson Noise: " + "lam = " + str(lam))
    # print("Salt and Pepper Noise: " + "Noise intensity = " + str(noise_intensity))

