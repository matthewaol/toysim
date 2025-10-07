
from toysim import lattice, molecule, older, bgnoise
import numpy as np


def produce_image(pdb_file_name, Qs, num_cells, cell_size, a, degrees):
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
    Tu = lattice.create_Tu_vectors(num_cells, cell_size)

    theta = degrees * np.pi / 180
    atoms = molecule.get_coords_pdb(pdb_file_name, "temporary_id")

    I_list = older.get_I_values_no_loop(Qs, atoms, Tu, f_j, theta)
    I_list_background = bgnoise.add_background_exp_no_loop(I_list, Qs, a)

    I_list_size = int(np.sqrt(len(I_list)))

    square_I_list = np.reshape(I_list_background, (I_list_size, I_list_size))

    return square_I_list