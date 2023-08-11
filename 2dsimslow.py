import simulation as sim 
import bgnoise as bg
import molecule
import lattice
import older

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

def get_coords_pdb(file_name, structure_id,get_only_xy=True):
    '''
    Gets atom coordinates from a .pdb file & applies symmetry operators
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

    # space_group = find_sym_info(file_name)
    # trans_ops, rot_ops = get_sym_operators(space_group)

    atoms = structure.get_atoms()

    list_of_coords = np.array([atom.get_coord() for atom in atoms])

    # M = get_M(file_name)

    # fractional_coords = np.dot(list_of_coords, M)
    
    # print("Applying symmetry operators")

    # all_coords = []
    # for t, r in zip(trans_ops,rot_ops): 
    #     x = np.dot(fractional_coords,r) + t
    #     big_x = np.dot(x, np.linalg.inv(M))
    #     all_coords.append(big_x)

    # all_coords = np.vstack(all_coords)

    return list_of_coords

if __name__ == '__main__':  
    start = time.time()
  
    print("Bringing in Qs from detector")
    beam = BeamFactory.from_dict(models.beam) 
    detector = DetectorFactory.from_dict(models.pilatus)

    qvecs, img_sh = detectors.qxyz_from_det(detector, beam)
    qvecs = qvecs[0] # specifies q-vectors     Tu = sim.create_Tu_vectors(20)

    M = molecule.get_M("4bs7.pdb")
    Mi = np.linalg.inv(M)
    a,b,c = Mi.T

    Tu = lattice.create_Tu_vectors_3d_basis(5,a,b,c)

    atoms = get_coords_pdb("4bs7.pdb", "temp", False)#[:200]

    deg = 60
    theta = deg / 180*np.pi
    Qs_size = int(np.sqrt(len(qvecs)))
    I_list = older.get_I_values(qvecs,atoms,Tu,1,theta)
    
    square_I_list = np.reshape(I_list, (Qs_size, Qs_size))

    end = time.time()
    print("Program took", str(end-start), "secs or", ((end-start)/60 ), "minutes") 
    print("image size: ", Qs_size, "by", Qs_size)
    print("# of pixels", len(qvecs))
    print('# of unit cells', len(Tu))

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', labelsize=30)  # Adjust the labelsize as needed
    
    plt.imshow(square_I_list, vmax=100)
    plt.show() 
    
    from IPython import embed;embed()

