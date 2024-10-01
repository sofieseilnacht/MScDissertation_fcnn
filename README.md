# Galaxy Ellipticity Prediction with CNNs in the Fourier Space

### Overview
This project aims to predict galaxy ellipticity values from radio interferometric data, which is captured in Fourier space, using a Convolutional Neural Network (CNN). The data was simulated using the gal_sim repository (forked from itrharrison's rwl_sims repository), and custom preprocessing steps were applied to format the data for model training.


### Table of Contents
1. [Overview](#overview)
2. [Simulation Requirements](#simulation-requirements)
3. [FCNN Requirements](#fcnn-requirements)
4. [Project Workflow](#project-workflow)
5. [Preprocessing](#preprocessing) 
6. [Model Training](#model-training)
7. [Results](#results)


### Environment Setup
All credit goes to Dr. Ian Harrison and his rwl_sims GitHub repo (forked in my own repository) for the creation of the simulations portion of this project. All files related to simulation creation (including dependencies) can be found in the Simulation directory.

This project was developed in virtual environments. To set up the environments and install the necessary dependencies, you can use the provided `.yml` files. Follow these steps:

1. Create a Virtual Environment (optional but recommended):

    For galaxy simulations:
    
        conda env create -f galsim.yml
        conda activate galsim
        
    For FCNN:

        conda env create -f simuclass.yml
        conda activate simuclass

2. Install Dependencies without a Virtual Environment: 

If you prefer not to create a virtual environment, you can use the requirements.txt files to install the necessary dependencies directly:

    For galaxy simulations:
        pip install -r gal_sim_requirements.txt

    For FCNN:

        pip install -r requirements.txt


Important Packages for galaxy simulations and the FCNN:
- numpy
- pandas
- astropy
- matplotlib
- scikit-learn
- TensorFlow
- Keras

Ensure you have Python installed (version 3.12.4 or later) to run the code.


### Project Workflow
1. **Simulation of Galaxies**:
   - Galaxies were simulated using the `gal_sim` repository. The simulations generated images of single galaxies with varying ellipticity, noise levels, and other astrophysical parameters. The ellipticity values range from 0.0 to 0.4, providing a dataset of thousands of galaxies with known ellipticity.

   - For this project, we modified the following within the `test.ini` file in the `inis` directory:
     ```ini
     [pipeline]
     output_suffix = test_mod_e0.0
     output_path = /Users/sofie/Desktop/Projects/rwl_sims/ellip_txts
     figure_path = /Users/sofie/Desktop/Projects/rwl_sims/ellip_images

     [skymodel]
     catalogue_filepath = /Users/sofie/Desktop/Projects/rwl_sims/data/catalogue_SFGs_complete_v4.1.fits.txt
     psf_filepath = /Users/sofie/Desktop/Projects/rwl_sims/data/ska1_mid_uniform.psf.fits
     ngals = 100
     constant_mod_e_value = 0.0
     ```
     We adjusted the `constant_mod_e_value` to values between 0.0 and 0.4, updating the `output_suffix` accordingly. We also made sure to update the paths accordingly.

   - To run the code and produce the simulations, execute the following in the terminal:
     ```bash
     python single_gaussians_galaxies.py inis/test.ini
     ```

   - For automated simulations, consider pulling the automated version from the `rwl_sims` repository, which cycles through values automatically. Note that file names and paths may need to be adjusted in the `preprocessing.py` file located in the `Final` directory of this repository.

2. **Preprocessing**:
The preprocessing step is crucial for preparing the data for training the Convolutional Neural Network (CNN). This involves two main tasks:

    - Applying Fourier Transform: The pixel data from simulated galaxies undergoes a Fourier transform, converting spatial domain data into frequency domain data. This transformation helps highlight the essential features of the galaxy images for better model performance.
    - Extracting Ellipticity Values: Corresponding ellipticity values are extracted from truth catalog FITS files. These values are vital as they serve as the target outputs for the CNN training process.

The script responsible for this preprocessing is located in the Final directory and is labeled preprocessing.py. Before running the script, ensure you adjust the paths to the input files. You can find the file paths specified at the top of the script:
    
        images_folder_path = '/path/to/ellip_images_test'
        truthcat_folder_path = '/path/to/ellip_txts'
        pickle_file_path = '/path/to/final_pickles'
    
Make sure to update these paths to reflect the correct locations of your data files. It is recommended to keep the file names the same as those in the original script to facilitate running other scripts related to the FCNN.