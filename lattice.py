import numpy as np


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
            3D numpy array of lattice vectors
    '''
    Tu = []

    for x in range(num_cells):  # Creates Tu vectors ranging from [0,0] to [Tu_size,Tu_size]
        for y in range(num_cells):
            for z in range(num_cells):
                Tu.append([x * cell_size, y * cell_size, z * cell_size])
    return np.array(Tu)


def create_Tu_vectors_3d_basis(num_cells, a_vec, b_vec, c_vec):
    ...
    # Tu_vec = x* a_vec + y* b_vec + z*c_vec
    # a_vec, b_vec, c_vec are 1d numpy arrays
    Tu = []
    for x in range(num_cells):  # Creates Tu vectors ranging from [0,0] to [Tu_size,Tu_size]
        for y in range(num_cells):
            for z in range(num_cells):
                Tu.append([x * a_vec + y * b_vec + z * c_vec])

    return np.array(Tu)[:, 0, :]


def create_Tu_vectors_3d_tetra(num_cells, a, c):
    Tu = []

    for x in range(num_cells):
        for y in range(num_cells):
            for z in range(num_cells):
                Tu.append([x * a, y * a, z * c])
    return np.array(Tu)


def create_Tu_vectors_3d_ortho(num_cells, a, b, c):
    Tu = []
    for x in range(num_cells):
        for y in range(num_cells):
            for z in range(num_cells):
                Tu.append([x * a, y * b, z * c])
    return np.array(Tu)
