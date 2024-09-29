import os
import numpy as np
from numpy import fft
from astropy.io import fits
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import re


# Define paths for folder access
images_folder_path = '/Users/sofie/Desktop/Projects/rwl_sims/ellip_images_test'
truthcat_folder_path = '/Users/sofie/Desktop/Projects/rwl_sims/ellip_txts'
pickle_file_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/final_pickles'


# Define a custom sorting function to sort the files by the ellipticity value defined within the file name
def sort_numerically(filenames):
    def numeric_key(filename):
        # Match the floating-point number at the beginning and the integer at the end
        match = re.search(r'(\d+\.\d+)ellipticity_(\d+)', filename)
        if match:
            return (float(match.group(1)), int(match.group(2)))
        return (float('inf'), float('inf'))  # In case there's no match, push it to the end

    return sorted(filenames, key=numeric_key)


# Preprocess pixel data by applying a Fourier transform, extract ellipticity values from truth catalog FITS files, and match the transformed data with corresponding ellipticity values
def preprocess_batch(pixel_batch, truthcat_folder_path):
    fft_batch_dict = {}
    ellip_batch_dict = {}

    for item in sort_numerically(pixel_batch):
        if item.endswith('.txt'):
            pixel_path = os.path.join(truthcat_folder_path, item)
            with open(pixel_path, 'r') as f:
            # Assuming the file contains comma-separated pixel values
                pixel_data = f.read()

                 # Convert to float and reshape into a 2D array
                pixel_values = np.array([float(x) for x in pixel_data.split()])
                
                # Determine the shape of the original image
                # You need to know the width and height; assuming they are known
                height, width = 199, 199  # Dimensions from the created galaxy simulations
                pixel_values = pixel_values.reshape((height, width))

                # Fourier transform the image
                fourier_pixel_data = fft.fftshift(fft.fft2(pixel_values))

            # Store the fft data in a dictionary
            if item not in fft_batch_dict:
                fft_batch_dict[item] = {}
            fft_batch_dict[item]['fourier_pixel_data'] = fourier_pixel_data  # Match key name

    # Process truth catalog FITS files
    for truthcat_file in os.listdir(truthcat_folder_path):
        if truthcat_file.startswith('truthcat'):
            truthcat_file_path = os.path.join(truthcat_folder_path, truthcat_file)
            truthcat_fits = fits.open(truthcat_file_path)
            hdr_data = truthcat_fits[1].data

            # Grab mod_e values
            ellipticity = hdr_data['mod_e']

            # Add values to ellipticity value dictionary
            if truthcat_file not in ellip_batch_dict:
                ellip_batch_dict[truthcat_file] = {}
            ellip_batch_dict[truthcat_file] = ellipticity

    # Match text with ellipticity data
    for fft_key in fft_batch_dict.keys():
        fft_prefix = fft_key[:7]
        fft_index = int(fft_key.split('ellipticity_')[1].split('.txt')[0])
        for ellip_key, ellip_value in ellip_batch_dict.items():
            ellip_prefix = ellip_key[9:16]  
            if fft_prefix == ellip_prefix:
                ellip_index = ellip_value[fft_index]
                fft_batch_dict[fft_key]['ellipticity'] = ellip_index
    return fft_batch_dict, ellip_batch_dict


# Define a function to dump all of the data into a pickle file
def save_to_pickle(data, pickle_file_path):
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data, f)


# Process batches of simulated image pixel data and ellipticity values, apply the Fourier transform, and save the processed batches to pickle files
def process_and_save_batches(truthcat_folder_path, pickle_file_path, batch_size=202):
    # Get the list of image filenames
    txt_filenames = sort_numerically([f for f in os.listdir(truthcat_folder_path) if f.endswith('.txt')])

    # Process and save batches
    for i in range(0, len(txt_filenames), batch_size):
        batch = txt_filenames[i:i+batch_size]
        fft_batch, ellip_batch = preprocess_batch(batch, truthcat_folder_path)
        batch_pickle_file_path = os.path.join(pickle_file_path, f'batch_{i//batch_size}.pkl')
        save_to_pickle((fft_batch, ellip_batch), batch_pickle_file_path)


# Define a function to load in the data from the pickle files
def load_data_from_pickle(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        return pickle.load(f)


# Organize fft-transformed image data and ellipticity values into input-output pairs for training, separating the real and imaginary parts of the Fourier data
def prepare_data_for_training(input_output_data_dict):
    input_output_data_list = []

    # Populate the list with data from the dictionaries
    for key, value in input_output_data_dict.items():
        # Extract fourier_image_data value and ellipticity value from the nested dictionaries
        fft_data = value.get('fourier_data', None)
        ellipticity = value.get('ellipticity', None)
        
        # Ensure both values are present before appending to input_output_data_array
        if fft_data is not None and ellipticity is not None:
            input_output_data_list.append((fft_data, ellipticity))

    # Append the data to lists to make it easier to deal with
    real_input = []
    imag_input = []
    output_e_data = []

    for data_tuple in input_output_data_list:
        fourier_image_data = data_tuple[0]
        ellipticity = data_tuple[1]
        
        input_fft_data_real.append(np.real(fourier_image_data))
        input_fft_data_imag.append(np.imag(fourier_image_data))
        output_e_data.append(ellipticity)

    # Convert the data to arrays:
    input_fft_data_real = np.array(input_fft_data_real)
    input_fft_data_imag = np.array(input_fft_data_imag)
    output_e_data = np.array(output_e_data)
    print(output_e_data)

    return real_input, imag_input, output_e_data


# Run the function to process the pixel data and ellipticity information from files and save it for later use
process_and_save_batches(truthcat_folder_path, pickle_file_path)

