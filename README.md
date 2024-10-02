# Galaxy Ellipticity Prediction with CNNs in the Fourier Space

## Overview
This project aims to predict galaxy ellipticity values from radio interferometric data, which is captured in Fourier space, using a linear regression Convolutional Neural Network (CNN). The data was simulated using the gal_sim repository (forked from itrharrison's rwl_sims repository), and custom preprocessing steps were applied to format the data for model training.


## Table of Contents
1. [Overview](#overview)
2. [Environment Setup](#environment-setup)
3. [Galaxy Simulations](#galaxy-simulations)
4. [Preprocessing](#preprocessing) 
5. [Model Training](#model-training)
6. [Results](#results)
7. [Licensing](#licensing)
8. [Open Source Contribution](#open_source_contribution)
9. [Contributors and Acknowledgements](#contributors_and_acknowledgements)


## Environment Setup
Credit goes to Dr. Ian Harrison and his rwl_sims GitHub repository for the creation of the simulations portion of this project. All files related to simulation creation (including dependencies) can be found in the Simulation directory.

This project was developed in virtual environments. To set up the environments and install the necessary dependencies, you can use the provided `.yml` files. Follow these steps:

1. Create a Virtual Environment (optional but recommended):

    - For galaxy simulations:
    
            conda env create -f galsim.yml
            conda activate galsim
        
    - For FCNN:

            conda env create -f simuclass.yml
            conda activate simuclass

2. Install Dependencies without a Virtual Environment: 

    If you prefer not to create a virtual environment, you can use the requirements.txt files to install the necessary dependencies directly:

    - For galaxy simulations:
    
            pip install -r gal_sim_requirements.txt

        
    - For FCNN:

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


## Simulation of Galaxies
   - Galaxies were simulated using the `gal_sim` repository. The simulations generated images of single galaxies with varying ellipticity, noise levels, and other astrophysical parameters. The ellipticity values range from 0.0 to 0.4, providing a dataset of thousands of galaxies with known ellipticity.

   - For this project, we modified the following within the `test.ini` file in the `inis` directory:
     ```ini
     [pipeline]
     output_suffix = test_mod_e0.0
     output_path = /path/to/ellip_txts
     figure_path = /path/to/ellip_images

     [skymodel]
     catalogue_filepath = /path/to/catalogue_SFGs_complete_v4.1.fits.txt
     psf_filepath = /path/to/ska1_mid_uniform.psf.fits
     ngals = 100
     constant_mod_e_value = 0.0
     ```
     We adjusted the `constant_mod_e_value` to values between 0.0 and 0.4, updating the `output_suffix` accordingly. We also made sure to update the paths accordingly.

   - To run the code and produce the simulations, execute the following in the terminal:
     ```bash
     python single_gaussians_galaxies.py inis/test.ini
     ```

   - For automated simulations, consider pulling the automated version from the `rwl_sims` repository, which cycles through values automatically. Note that file names and paths may need to be adjusted in the `preprocessing.py` file located in the `Final` directory of this repository.


## Preprocessing
In this project, we preprocess simulated galaxy data to prepare it for training the Convolutional Neural Network (CNN). This step is crucial and involves the following tasks:

   - File Organization: Data files are organized in specified directories for easy access.

   - Fourier Transform: We apply a Fast Fourier Transform (FFT) to pixel data, reshaping it into 2D arrays for processing. This transformation helps highlight the essential features of the galaxy images for better model performance.

   - Ellipticity Extraction: Corresponding ellipticity values are extracted from truth catalog FITS files. These values are vital as they serve as the target outputs for the CNN training process.

   - Batch Processing: Data is processed in batches, and results are saved in pickle files for efficient loading during training.

The script responsible for this preprocessing is located in the Final directory and is labeled preprocessing.py. Before running the script, ensure you adjust the paths to the input files:
  
        images_folder_path = '/path/to/ellip_images_test'
        truthcat_folder_path = '/path/to/ellip_txts'
        pickle_file_path = '/path/to/final_pickles'
    
Make sure to update these paths to reflect the correct locations of your data files. It is recommended to keep the file names the same as those in the original script to facilitate running other scripts related to the FCNN.

For detailed implementation, see the 'process_and_save_batches' and 'preprocess_batch' functions in the code.


## Model Training
This section outlines the training process for the Convolutional Neural Network (CNN) utilized to predict galaxy ellipticity values. The relevant code can be found in the `fcnn.py` file located in the `Final` directory. This script showcases the best-performing model architecture, having been optimized through experimentation with over ten different architectures, where various metrics were employed to identify the most effective model. An example of our second-best performing model architecture, along with the modifications made during the optimization process, can be found in the 'Archive' directory.


1. **Model Architecture**: The model is designed as a linear regression Convolutional Neural Network (CNN) to analyze processed galaxy images in Fourier space. Its architecture comprises multiple convolutional layers followed by dense layers, enabling it to capture intricate features and relationships within the data. A visual representation of this architecture is available in the `Project_Results/plots` directory under the filename `model_visualization.png`.

2. **Training Procedure**:
   - The training process utilizes datasets that have been preprocessed and saved as pickle files.

   - The model is trained with mean squared error (MSE) as the loss function, employing the Adam optimizer for enhanced performance.

   - Key training metrics include Mean Absolute Error (MAE), MSE, and Root Mean Squared Error (RMSE) to evaluate the model's accuracy.

3. **Training Command**:
   To initiate the training process, execute the following command in your terminal:
   ```bash
   python train_model.py
   ```

Ensure the script paths and filenames reflect your setup.

4. **Hyperparameters**:
    - Learning Rate: 0.001
    - Batch Size: 32
    - Epochs: 100

5. **Evaluation**: Upon completion of training, the model is evaluated using a separate validation dataset to assess its performance. The evaluation metrics include normalized and non-normalized MAE, MSE, RMSE, and an astrophysical difference metric that quantifies the predicted values against the true labels, facilitating a comprehensive analysis of the model's performance. These metrics are automatically output to .csv files when the script is run.


## Results
The findings of this project and a discussion of future work possibilities are detailed in the 'final_dissertation.pdf' file located in the 'Dissertation' directory.

Additionally, the 'difference_metrics.py' script, found in the 'Final' directory, generates metrics derived from the model's predictions. This script processes CSV files containing true and predicted values for training, validation, and test datasets across epochs, calculating the differences between these values. It organizes the data by epoch and saves the results into new CSV files, facilitating a comprehensive analysis of model performance over time.

As mentioned in the sections above, ensure the file paths are updated when running each code.

For visual representation, the 'true_pred_graph.py' script in the same directory allows us to output these metrics for comparison. To view the plots associated with the project results, please refer to the 'Project_Results/plots' directory.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Open Source Contribution
This project is open-source, and contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, feel free to create a pull request or open an issue. Please ensure that your contributions align with the project's goals and follow the code of conduct outlined in the repository.


## Contributors and Acknowledgements
I would like to express my heartfelt gratitude to Dr. Ian Harrison for his invaluable supervision throughout this project. His innovative vision and the code for the galaxy simulations laid the groundwork for my work, allowing me to take charge of the project.

A special thank you to Dr. Martin Chorley and his PhD student, Sam Everest, for their insightful suggestions on preprocessing and data management, which significantly optimized RAM usage and improved efficiency.

I also extend my appreciation to Jenny Highfield and the Cardiff University Cyber Security Defense Lab for their generous support in providing the essential computing resources that enabled the completion of this experiment.

Furthermore, I am thankful to Dr. Oktay Karakus and his PhD students, Zien Ma and Wanli Ma, for their expert guidance on model architecture in Fourier space using complex values. Their contributions were instrumental to the success of this project.
