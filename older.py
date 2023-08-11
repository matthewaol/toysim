import numpy as  np
import numexpr as ne


# Molecular Transform functions
def molecular_transform(Q, Atoms, f_j,
                        theta):  # takes in a single Q vector and a list of atoms, outputs a single A value
    a = b = 0

    for atom in Atoms:
        phase = np.dot(Q, atom)
        a += np.cos(phase)
        b += np.sin(phase)

    i_real, i_imag = a * f_j, b * f_j
    return i_real + i_imag * 1j


def molecular_transform_no_loop(Q, Atoms, f_j,
                                theta):  # takes in a single Q vector and a list of atoms, outputs a single A value
    a = b = 0
    rotation_m = [[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]

    rotated_u = np.dot(rotation_m, Atoms.T)
    phase = np.dot(Q, rotated_u)
    a = sum(np.cos(phase))
    b = sum(np.sin(phase))
    i_real, i_imag = a * f_j, b * f_j
    return i_real + i_imag * 1j


# assign one unit/thread/worker one pixel, do the q-vec cos/sin computation in the gpu
# instead of being 8k * 6m operations, it'll be 8k operations

def molecular_transform_no_loop_array(Qs, Atoms, f_j, theta):  # takes in an array of Q instead of a single Q
    a = b = 0
    rotation_m = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])

    rotated_u = np.dot(rotation_m, Atoms.T)
    phase = np.dot(Qs, rotated_u)
    a = np.sum(np.cos(phase), axis=1)  # indicating axis = 1, to sum over the columns instead of rows
    b = np.sum(np.sin(phase), axis=1)
    i_real, i_imag = a * f_j, b * f_j
    return i_real + i_imag * 1j


def molecular_transform_no_loop_array_3d(Qs, Atoms, rotation_m):
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

    Qs_mag = np.linalg.norm(Qs, axis=1)
    exp_arg = Qs_mag ** 2 / 4 * -10.7
    f_j = 7 * np.exp(exp_arg)

    rotated_u = np.dot(rotation_m, Atoms.T)
    phase = np.dot(Qs, rotated_u)

    cos_phase = ne.evaluate("cos(phase)")
    sin_phase = ne.evaluate("sin(phase)")

    a = np.sum(cos_phase, axis=1)
    b = np.sum(sin_phase, axis=1)

    i_real, i_imag = a * f_j, b * f_j

    return i_real + i_imag * 1j


def molecular_transform_chunks(Qs, Atoms, rotation_m, chunk_size):
    a2 = np.zeros(len(Qs))
    b2 = np.zeros(len(Qs))

    Qs_mag = np.linalg.norm(Qs, axis=1)
    exp_arg = Qs_mag ** 2 / 4 * -10.7
    f_j = 7 * np.exp(exp_arg)

    num_chunks = len(Atoms) / chunk_size

    print("Chunking up atoms")
    chunked_atoms = (np.array_split(Atoms, num_chunks))
    print("Computing molecular transform for each chunk")

    i = 0
    for chunk in chunked_atoms:
        i += 1
        print("Computing chunk #", i)

        rotated_u = np.dot(rotation_m, chunk.T)
        chunk_phase = np.dot(Qs, rotated_u)

        a2 += np.sum(ne.evaluate("cos(chunk_phase)"), axis=1)
        b2 += np.sum(ne.evaluate("sin(chunk_phase)"), axis=1)

    i_real, i_imag = a2 * f_j, b2 * f_j
    return i_real + i_imag * 1j


def test_molecular_transform():
    '''
    Testing the molecular transform withh the chunking version
    '''
    test_qs = np.random.random((5000, 3))  # create_q_vectors_3d(20,.1)
    test_atoms = np.random.random((1000, 3))  # np.array([[3,4,1],[3,1,6],[2,1,5],[3,2,1]])
    rand_rot_mat = sp.spatial.transform.Rotation.random(1, random_state=0)
    rotat_mat = rand_rot_mat.as_matrix()[0]
    molecular = molecular_transform_no_loop_array_3d(test_qs, test_atoms, rotat_mat)
    molecular_chunks = molecular_transform_chunks(test_qs, test_atoms, rotat_mat, 100)
    assert np.allclose(molecular, molecular_chunks)
    print("Molecular transform passed!")


# Lattice transform functions
def lattice_transform(Q, Tu, theta):  # takes in a single Q vector and a list of Tu vectors, outputs a single A value
    a = b = 0

    for u in Tu:
        # u is the vector representing the distance between the origin and the corner of a lattice unit
        phase = np.dot(Q, u)
        a += np.cos(phase)
        b += np.sin(phase)
    i_real, i_imag = a, b
    return i_real + i_imag * 1j


def lattice_transform_no_loop(Q, Tu,
                              theta):  # takes in single Q vector and list of Tu vectors, outputs a single A value
    rotation_m = [[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]

    rotated_u = np.dot(rotation_m, Tu.T)  # Applying rotation to array of Tu vectors
    phase = np.dot(Q, rotated_u)  # applying dot product to a Q vector and rotated Tu vectors
    a = sum(np.cos(phase))  # cosine of phase
    b = sum(np.sin(phase))  # sin of phase
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

    rotated_u = np.dot(rotation_m, Tu.T)
    phase = np.dot(Qs, rotated_u)
    a = np.sum(np.cos(phase), axis=1)
    b = np.sum(np.sin(phase), axis=1)

    i_real, i_imag = a, b
    return i_real + i_imag * 1j


def lattice_transform_no_loop_array_3d(Qs, Tu, rotation_m):  # 3d Qs & Tu, takes in rotation matrix
    a = b = 0

    rotated_u = np.dot(rotation_m, Tu.T)
    phase = np.dot(Qs, rotated_u)

    cos_phase = ne.evaluate("cos(phase)")
    sin_phase = ne.evaluate("sin(phase)")

    a = np.sum(cos_phase, axis=1)  # indicating axis = 1, to sum over the columns instead of rows
    b = np.sum(sin_phase, axis=1)

    i_real, i_imag = a, b
    return i_real + i_imag * 1j


# Complete transform functions
def get_I_values(Qs, Atoms, Tu, f_j,
                 theta):  # calculating the lattice and molecular transforms and returning intensities

    print("Computing intensities")

    I_list = []
    for Q in Qs:  # for each Q value do these steps!
        a_molecular = molecular_transform(Q, Atoms, f_j, theta)  # apply molecular transform!
        a_lattice = lattice_transform(Q, Tu, theta)  # apply lattice transform!
        a_total = a_molecular * a_lattice  # multiply them together
        final_I = a_total.real ** 2 + a_total.imag ** 2  # add the two complex numbers
        I_list.append(final_I)  # append that to list

    print("Finished intensities")

    return np.array(I_list)


def get_I_values_no_loop(Qs, Atoms, Tu, f_j, theta):
    print("Computing intensities")

    a_molecular = molecular_transform_no_loop_array(Qs, Atoms, f_j, theta)
    a_lattice = lattice_transform_no_loop_array(Qs, Tu, theta)
    a_total = a_molecular * a_lattice

    print("Finished intensities")
    return a_total.real ** 2 + a_total.imag ** 2


def get_I_values_no_loop_3d(Qs, Atoms, Tu, rotation_m):
    print("Computing intensities")

    a_molecular = molecular_transform_no_loop_array_3d(Qs, Atoms, rotation_m)
    a_lattice = lattice_transform_no_loop_array_3d(Qs, Tu, rotation_m)
    a_total = a_molecular * a_lattice

    print("Finished intensities")
    return a_total.real ** 2 + a_total.imag ** 2


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

    for x in np.arange(-image_size, image_size + .01, step_size):
        for y in np.arange(-image_size, image_size + .01, step_size):
            q_vectors.append([x, y])
    return np.array(q_vectors)


def create_q_vectors_3d(image_size, step_size):
    q_vectors = []

    for x in np.arange(-image_size, image_size + .01, step_size):
        for y in np.arange(-image_size, image_size + .01, step_size):
            q_vectors.append([x, y, np.sqrt(x ** 2 + y ** 2)])
    return np.array(q_vectors)