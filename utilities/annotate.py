import scipy as sp


def count_spots(square_I_list, threshold_factor=None):
    if threshold_factor is not None:
        threshold = square_I_list > threshold_factor
    else:
        threshold = square_I_list

    labels, num_of_labels = sp.ndimage.label(
        threshold)  # Label each index of intensity array based on our defined threshold
    peaks = sp.ndimage.find_objects(labels)  # gets tuples corresponding to the location of each peak
    print("Number of spots: " + str(num_of_labels))
    return num_of_labels, peaks