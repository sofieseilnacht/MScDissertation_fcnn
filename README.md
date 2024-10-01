# Galaxy Ellipticity Prediction with CNNs in the Fourier Space

### Overview
This project aims to predict galaxy ellipticity values from radio interferometric data, which is captured in Fourier space, using a Convolutional Neural Network (CNN). The data was simulated using the gal_sim repository (forked from itrharrison's rwl_sims repository), and custom preprocessing steps were applied to format the data for model training.


### Table of Contents
tbd

### Requirements 
This project was developed in a virtual environment. To set up the environment and install the necessary dependencies, you can use the provided .yml file. Follow these steps:

1. Create a Virtual Environment (optional but recommended):
    
    conda env create -f simuclass.yml
    conda activate simuclass
        
2. Install Dependencies without a Virtual Environment: 
If you prefer not to create a virtual environment, you can use the requirements.txt file to install the necessary dependencies directly:

    pip install -r requirements.txt


### Project Workflow
1. Simulation of Galaxies:
Galaxies were simulated using the gal_sim repository. The simulations generated simulations of single galaxy images with varying ellipticity, noise levels, and other astrophysical parameters. The ellipticity values range from 0.0 to 0.4, providing a dataset of thousands of galaxies with known ellipticity. 

2. Preprocessing:
The preprocessing step involves applying a Fourier transform to the pixel data of simulated galaxies and extracting the corresponding ellipticity values from truth catalog FITS files. This process prepares the data for training the Convolutional Neural Network (CNN). The script for this can be found in the "Final" directory and is labeled 