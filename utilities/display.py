import pylab as plt


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

    photo = plt.imshow(square_I_list, vmax=vmax, vmin=vmin)
    plt.show()