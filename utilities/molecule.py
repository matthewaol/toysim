import scipy as sp
from iotbx import pdb
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from cctbx import sgtbx


def determine_cell_size(atoms, scale=5):
    '''
    Determines an appropriate cell size for lattice based on the maximum distance value from array of atoms
        Parameters:
            atoms: Array of atom coordinates
            scale: scale the size of the proteing by this amount to set cell size
        Returns: float value of the maximum distance found in array of atom coordinates
    '''
    return sp.spatial.distance.pdist(atoms).max() * scale


def find_sym_info(file_name): # returns space group info
    P = pdb.input(file_name)
    symbol = P.crystal_symmetry().space_group_info().type().lookup_symbol()
    return symbol


def get_M( pdb_file):
    P = pdb.input(pdb_file)
    uc = P.crystal_symmetry().unit_cell()
    Mi = np.reshape (uc.orthogonalization_matrix(), (3,3))
    M = np.linalg.inv(Mi)
    return M


def get_sym_operators(space_group_info): # gets arrays of sym operators
    sg = sgtbx.space_group_info(space_group_info)
    gr = sg.group()
    Ops = gr.all_ops()
    trans = [np.reshape(O.t().as_double(), (1,3)) for O in Ops]
    rots = [np.reshape(O.r().as_double(), (3,3)) for O in Ops]

    return np.array(trans), np.array(rots)


def get_coords_pdb(file_name, structure_id, get_only_xy=False, apply_sym_ops=True, skip_hydrogen=True):
    '''
    Gets atom coordinates from a .pdb file & applies symmetry operators
    Parameters:
        file_name: string path to file
        structure_id: string id of structure
        get_only_xy: Assign false if xyz coordinates are needed
        apply_sym_ops: apply translation and rotation symmetry operators
        skip_hydrogen: only extract the non-hydrogen atom coordinates
    Returns:
        numpy array of atom coordinates
    '''
    print("Getting atom coordinates from molecule")
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(structure_id, file_name)

    atoms = list(structure.get_atoms())
    if skip_hydrogen:
        atoms = [a for a in atoms if a.element != 'H']

    list_of_coords = np.array([atom.get_coord() for atom in atoms])
    list_of_coords = list_of_coords
    print("Atoms shape:", list_of_coords.shape)

    if apply_sym_ops:
        space_group = find_sym_info(file_name)
        trans_ops, rot_ops = get_sym_operators(space_group)
        M = get_M(file_name)
        fractional_coords = np.dot(list_of_coords, M)

        print("Applying symmetry operators")

        all_coords = []
        for t, r in zip(trans_ops,rot_ops):
            x = np.dot(fractional_coords,r) + t
            big_x = np.dot(x, np.linalg.inv(M))
            all_coords.append(big_x)

        list_of_coords = np.vstack(all_coords)

    if get_only_xy:
        list_of_coords = np.array(np.delete(list_of_coords, 2, axis=1))

    return list_of_coords
