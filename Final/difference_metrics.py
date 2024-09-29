import os
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import rc
import pickle
import re
import pdb
import seaborn as sns


# Define text settings for graphs
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)
plt.close('all') # tidy up any unshown plots


# Define your paths below
run1_folder_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/predictions_run1'


# Define a function to sort through the csv file names (containing the metrics per epoch) based upon the epoch number
def sort_numerically(filenames):
    def numeric_key(filename):
        # Match the floating-point number at the beginning and the integer at the end
        match = re.search(r'epoch_(\d+)', filename)
        if match:
            return (float(match.group(1)), int(match.group(2)))
        return (float('inf'), float('inf'))  # In case there's no match, push it to the end
    return sorted(filenames, key=numeric_key)


# Clean up the metrics by removing unnecessary characters and turn the true data into a float
def clean_and_convert(value):
    # Remove brackets from the string and convert to float.
    try:
        # Remove brackets and whitespace, then convert to float
        cleaned_value = value.strip('[]').strip()
        return float(cleaned_value)
    except ValueError:
        # If conversion fails, return NaN
        return np.nan


# Turn the predicted data into a float
def is_numeric(value):
    # Attempt to convert the string to a float and return the result.If conversion fails, return NaN.
    try:
        return float(value)
    except ValueError:
        return np.nan


# Processes CSV files containing true and predicted values for training, validation, and test data across epochs, calculate the differences between true and predicted values, organize the data by epoch, and save the results into new CSV files for each epoch
def process_csvs(yT_yP_folder_path):
    y_true_train_values_by_epoch = {}
    y_pred_train_values_by_epoch = {}
    y_true_val_values_by_epoch = {}
    y_pred_val_values_by_epoch = {}
    y_true_test_values_by_epoch = {}
    y_pred_test_values_by_epoch = {}
    train_diff_by_epoch = {}
    val_diff_by_epoch = {}
    test_diff_by_epoch = {}

    for csv_file in os.listdir(yT_yP_folder_path):
        csv_path = os.path.join(yT_yP_folder_path, csv_file)
        
        # Extract the epoch number from the filename
        epoch_match = re.search(r'epoch(\d+)', csv_file)
        if epoch_match:
            print("epoch matcg")
            epoch_num = int(epoch_match.group(1))
        else:
            continue  # Skip files that don't have an epoch number

        if csv_file.endswith('train.csv'):
            print(f"Loading csv file: {csv_file}")
            
            try:
                df_train = pd.read_csv(csv_path, header=None)
                
                true_train_values = df_train.iloc[:, 0].values
                pred_train_values = df_train.iloc[:, 1].values
                
                true_train_values = [is_numeric(val) for val in true_train_values]
                pred_train_values = [clean_and_convert(val) for val in pred_train_values]
                
                # Store by epoch
                y_true_train_values_by_epoch[epoch_num] = true_train_values
                y_pred_train_values_by_epoch[epoch_num] = np.round(pred_train_values, 3)
                train_diff_by_epoch[epoch_num] = y_true_train_values_by_epoch[epoch_num] - y_pred_train_values_by_epoch[epoch_num]
            
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        elif csv_file.endswith('val.csv'):
            print(f"Loading csv file: {csv_file}")
            
            try:
                df_val = pd.read_csv(csv_path, header=None)
                
                true_val_values = df_val.iloc[:, 0].values
                pred_val_values = df_val.iloc[:, 1].values
                
                true_val_values = [is_numeric(val) for val in true_val_values]
                pred_val_values = [clean_and_convert(val) for val in pred_val_values]
                
                # Store by epoch
                y_true_val_values_by_epoch[epoch_num] = true_val_values
                y_pred_val_values_by_epoch[epoch_num] = np.round(pred_val_values, 3)
                val_diff_by_epoch[epoch_num] = y_true_val_values_by_epoch[epoch_num] - y_pred_val_values_by_epoch[epoch_num]
            
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        elif csv_file.startswith('test_'):
            print(f"Loading csv file: {csv_file}")
            
            try:
                df_test = pd.read_csv(csv_path, header=None)
                
                true_test_values = df_test.iloc[:, 0].values
                pred_test_values = df_test.iloc[:, 1].values
                
                true_test_values = [is_numeric(val) for val in true_test_values]
                pred_test_values = [clean_and_convert(val) for val in pred_test_values]
                
                # Store by epoch
                y_true_test_values_by_epoch[epoch_num] = true_test_values
                y_pred_test_values_by_epoch[epoch_num] = np.round(pred_test_values, 3)
                test_diff_by_epoch[epoch_num] = y_true_test_values_by_epoch[epoch_num] - y_pred_test_values_by_epoch[epoch_num]
            
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    # Convert dictionaries to DataFrames and save them to CSV files
    for epoch_num in sorted(y_true_train_values_by_epoch.keys()):
        print(f"Saving CSV for epoch {epoch_num}...")  # Debugging statement
        try:
            pd.DataFrame(train_diff_by_epoch[epoch_num]).to_csv(f"train_diff_epoch_{epoch_num}.csv", index=False, header=False)
            pd.DataFrame(val_diff_by_epoch[epoch_num]).to_csv(f"val_diff_epoch_{epoch_num}.csv", index=False, header=False)
            pd.DataFrame(test_diff_by_epoch[epoch_num]).to_csv(f"test_diff_epoch_{epoch_num}.csv", index=False, header=False)
            
            pd.DataFrame(y_true_train_values_by_epoch[epoch_num]).to_csv(f"true_train_epoch_{epoch_num}.csv", index=False, header=False)
            pd.DataFrame(y_pred_train_values_by_epoch[epoch_num]).to_csv(f"pred_train_epoch_{epoch_num}.csv", index=False, header=False)
            print(f"CSV saved successfully for epoch {epoch_num}.")  # Confirm the save
        except Exception as e:
            print(f"Error saving CSV for epoch {epoch_num}: {e}")

    print("Loading and separation by epoch complete.")

    return y_true_train_values_by_epoch, y_pred_train_values_by_epoch, y_true_val_values_by_epoch, y_pred_val_values_by_epoch, y_true_test_values_by_epoch, y_pred_test_values_by_epoch, train_diff_by_epoch, val_diff_by_epoch, test_diff_by_epoch


# Output the metric values to a csv file per epoch for the model run
run1 = (process_csvs(run1_folder_path), "Run 1")