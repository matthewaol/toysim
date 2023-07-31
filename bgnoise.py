import numpy as np

# Background functions

def add_background_water(I_list, Qs_magnitude, lower_bound=5, upper_bound=10, intensity=1e5):
    indices = np.where((Qs_magnitude >= lower_bound) & (Qs_magnitude <= upper_bound)) 

    for i in indices:
        I_list[i]+= intensity
    return I_list
    # If qs_mag == this range of values, 
    # save the indices 
    # then add a constant to the same indices in I_list


def add_background_exp(I_list, Qs, a): #add background based on exponential decay
    background_list = []
    
    for q in Qs: 
        x_comp, y_comp = q[0], q[1]
        B = np.exp(-(x_comp**2 + y_comp**2)* a)
        background_list.append(B)
        
    return np.array(background_list) * I_list

def add_background_exp_no_loop(I_list, Qs, a):
    B_list = np.exp(-(Qs[:,0]**2 + Qs[:,1]**2 )* a) * I_list
    return B_list 

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
