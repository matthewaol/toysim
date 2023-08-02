import numpy as np
import scipy as sp
# Background functions

def add_background_file(bg_file, Q_magnitude,intensity=1):
    '''
    Uses interpolation to layer background files over array of intensity values
        Parameters:
            bg_file: string path to background .txt file
            I_list: 1D array of image intensities (x-coords)
            Q_magnitude: 1D array of magnitudes from Q-vectors (y-coords)
            intensity: intensity of the background (default=100)
        Returns: 
            List of intensity values with background added to it 
    '''
    qmags, I_vals = np.loadtxt(bg_file).T
    #qmags = qmags *2 
    interpolated_func = sp.interpolate.interp1d(qmags, I_vals,fill_value="extrapolate")
    new_intensities = interpolated_func(Q_magnitude)

    print("Getting background")
    return new_intensities*intensity 
    
def add_background_water_offset(I_list, Q_magnitude, lower_bound=5, upper_bound=10, intensity=1e5):
    '''
    Adds donut shaped offset to image, centered at the origin
        Parameters: 
            I_list: 1D array of image intensities
            Q_magnitude: 1D array of magnitudes from Q-vectors
            lower_bound: inner radius of donut (default=5)
            upper_bound: outer radius of donut (default=10) 
            intensity: intensity of the offset (default=1e5)
        Returns: 
            List of intensities with donut offset added into it 
    '''
    indices = np.where((Q_magnitude >= lower_bound) & (Q_magnitude <= upper_bound)) 

    for i in indices:
        I_list[i]+= intensity
    return I_list

def add_background_exp(I_list, Qs, a): #add background based on exponential decay
    background_list = []
    
    for q in Qs: 
        x_comp, y_comp = q[0], q[1]
        B = np.exp(-(x_comp**2 + y_comp**2)* a)
        background_list.append(B)
        
    return np.array(background_list) * I_list

def add_background_exp_no_loop(I_list, Qs, a):
    B_list = np.exp(-(np.linalg.norm(Qs,axis=1))* a) * I_list
    return np.array(B_list) 

def add_background_offset(I_list,a): # adds constant offset to I_list
    return I_list + a

def add_background_gaussian(I_list, mu, sigma): # adds gaussian background to I_list / increase sigma for more background
    gaussian_background_list = np.random.default_rng().normal(mu,sigma,len(I_list))
    return I_list + gaussian_background_list 

def add_background_cauchy(I_list): #doesnt work yet
    cauchy_list = np.random.default_rng().standard_cauchy(len(I_list))
    return cauchy_list + I_list

# Noise functions

def add_gaussian_noise(I_list, mu, sigma): # returns the list of I's with gaussian noise multiplied into it 
    gaussian_array = np.random.default_rng().normal(mu, sigma, len(I_list)) 
    noisy_I_list = I_list * gaussian_array
    return noisy_I_list 

def add_poisson_noise(I_list,lam): # returns the list of I's with poisson noise multiplied into it 
    poisson_array = np.random.default_rng().poisson(lam, len(I_list))
    noisy_I_list = I_list * poisson_array
    return noisy_I_list

def add_saltpepper_noise(I_list,noise_level): #returns the list of I's with salt+pepper scattered in it randomly
    
    rng = np.random.default_rng()
    black_or_white = [0,255] # list containing color value for black or white 
    pixel_values = np.array(range((len(I_list)))) # list containing indices of I_list
    
    for i in range(noise_level): 
        rand_index = rng.choice(pixel_values,replace=False) #random generated number with the range of the I_list size 
        I_list[rand_index] = rng.choice(black_or_white)
    return I_list
