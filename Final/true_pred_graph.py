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


# Set text settings
rc('text', usetex=True)
rc('font', family='times')
rc('font', size=11)


# Define file paths for each run (each run contains a different model architecture) for metric comparison
best_run_folder_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/predictions_run4'
second_best_folder_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/predictions_run2'


# Define a function that custom sorts the files based on the epoch number in the file name
def sort_numerically(filenames):
    def numeric_key(filename):
        # Match the floating-point number at the beginning and the integer at the end
        match = re.search(r'epoch_(\d+)', filename)
        if match:
            return (float(match.group(1)), int(match.group(2)))
        return (float('inf'), float('inf'))  # In case there's no match, push it to the end
    return sorted(filenames, key=numeric_key)


# Cleans up the data (from csv file), removing any unnecessary characters and turning the data into floats
def clean_and_convert(value):
    # Remove brackets from the string and convert to float.
    try:
        # Remove brackets and whitespace, then convert to float
        cleaned_value = value.strip('[]').strip()
        return float(cleaned_value)
    except ValueError:
        # If conversion fails, return NaN
        return np.nan


# Turns a string (in the data) to a float
def is_numeric(value):
    # Attempt to convert the string to a float and return the result.If conversion fails, return NaN.
    try:
        return float(value)
    except ValueError:
        return np.nan


# Process CSV files containing true and predicted values for training, validation, and testing datasets, cleaning and converting the data before returning it as NumPy arrays
def process_csvs(yT_yP_folder_path):
    y_true_train_values = []
    y_pred_train_values = []
    y_true_val_values = []
    y_pred_val_values = []
    y_true_test_values = []
    y_pred_test_values = []

    for csv_file in os.listdir(yT_yP_folder_path):
        csv_path = os.path.join(yT_yP_folder_path, csv_file)
        
        if csv_file.endswith('train.csv'):  # Ensure that the file is a CSV
            print(f"Loading csv file: {csv_file}")
            
            try:
                df_train = pd.read_csv(csv_path, header=None)
                
                # Extract column values
                true_train_values = df_train.iloc[:, 0].values
                pred_train_values = df_train.iloc[:, 1].values
                
                # Process and clean the true values
                true_train_values = [is_numeric(val) for val in true_train_values]
                true_train_values = [val for val in true_train_values if not np.isnan(val)]
                
                # Process and clean the predicted values
                pred_train_values = [clean_and_convert(val) for val in pred_train_values]
                pred_train_values = [val for val in pred_train_values if not np.isnan(val)]
                
                # Round predicted values and append to the lists
                y_true_train_values.extend(true_train_values)
                y_pred_train_values.extend(np.round(pred_train_values, 3))
            
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        elif csv_file.endswith('val.csv'):  # Ensure that the file is a CSV
            print(f"Loading csv file: {csv_file}")
            
            try:
                df_val = pd.read_csv(csv_path, header=None)
                
                # Extract column values
                true_val_values = df_val.iloc[:, 0].values
                pred_val_values = df_val.iloc[:, 1].values
                
                # Process and clean the true values
                true_val_values = [is_numeric(val) for val in true_val_values]
                true_val_values = [val for val in true_val_values if not np.isnan(val)]
                
                # Process and clean the predicted values
                pred_val_values = [clean_and_convert(val) for val in pred_val_values]
                pred_val_values = [val for val in pred_val_values if not np.isnan(val)]
                
                # Round predicted values and append to the lists
                y_true_val_values.extend(true_val_values)
                y_pred_val_values.extend(np.round(pred_val_values, 3))
            
            except FileNotFoundError:
                print(f"File not found: {csv_path}")
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        elif csv_file.startswith('test_'):  # Ensure that the file is a CSV
            print(f"Loading csv file: {csv_file}")
            
            try:
                df_test = pd.read_csv(csv_path, header=None)
                
                # Extract column values
                true_test_values = df_test.iloc[:, 0].values
                pred_test_values = df_test.iloc[:, 1].values
                
                # Process and clean the true values
                true_test_values = [is_numeric(val) for val in true_test_values]
                true_test_values = [val for val in true_test_values if not np.isnan(val)]
                
                # Process and clean the predicted values
                pred_test_values = [clean_and_convert(val) for val in pred_test_values]
                pred_test_values = [val for val in pred_test_values if not np.isnan(val)]
                
                # Round predicted values and append to the lists
                y_true_test_values.extend(true_test_values)
                y_pred_test_values.extend(np.round(pred_test_values, 3))
            
            except FileNotFoundError:
                print(f"File not found: {csv_path}")
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    # Convert lists to NumPy arrays
    try:
        y_true_train_values = np.array(y_true_train_values, dtype=float)
        y_pred_train_values = np.array(y_pred_train_values, dtype=float)
        y_true_val_values = np.array(y_true_val_values, dtype=float)
        y_pred_val_values = np.array(y_pred_val_values, dtype=float)
        y_true_test_values = np.array(y_true_test_values, dtype=float)
        y_pred_test_values = np.array(y_pred_test_values, dtype=float)

    except ValueError as e:
        print(f"Error converting to NumPy array: {e}")
    
    print("Loading process complete.")

    return y_true_train_values, y_pred_train_values, y_true_val_values, y_pred_val_values, y_true_test_values, y_pred_test_values


# Calculate the average predicted values from CSV files containing true and predicted values for training, validation, and testing datasets, cleaning and processing the data before returning the averaged results as lists of arrays
def average_preds(yT_yP_folder_path):
    ttrain_values = []
    ptrain_values = []
    tval_values = []
    pval_values = []
    ttest_values = []
    ptest_values = []

    for file_csv in os.listdir(yT_yP_folder_path):
        file_path = os.path.join(yT_yP_folder_path, file_csv)
        
        if file_csv.endswith('train.csv'):  # Ensure that the file is a CSV
            print(f"Loading csv file: {file_csv}")
            
            try:
                df_train_ave = pd.read_csv(file_path)

                print("sorting df")
                df_train_ave_sorted = df_train_ave.sort_values(by = "True", ascending = True)

                print("Changing pred column to strings")
                df_train_ave_sorted['Predicted'] = df_train_ave_sorted['Predicted'].astype(str)
                df_train_ave_sorted['True'] = df_train_ave_sorted['True'].astype(str)


                print("Splicing the brackets")
                df_train_ave_sorted['Predicted'] = df_train_ave_sorted['Predicted'].map(lambda x: x.lstrip('[]').rstrip(']'))

                print("Changing columns to floats")
                df_train_ave_sorted['Predicted'] = df_train_ave_sorted['Predicted'].astype(float)
                df_train_ave_sorted['True'] = df_train_ave_sorted['True'].astype(float)

                print("averaging sorted df")
                df_train_averaged = df_train_ave_sorted.groupby(['True']).mean().reset_index()

                print("separating true and pred values")
                ttrain_dfvalues = df_train_averaged.iloc[:, 0].values
                ptrain_dfvalues = df_train_averaged.iloc[:, 1].values
                
                # print("Round the pred values if needed")
                ttrain_values.append(ttrain_dfvalues)
                ptrain_values.append(ptrain_dfvalues)     #np.round(ptrain_values, 3))
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")  
        
        elif file_csv.endswith('val.csv'):  # Ensure that the file is a CSV
            print(f"Loading csv file: {file_csv}")
            
            try:
                df_val_ave = pd.read_csv(file_path)

                print("sorting df")
                df_val_ave_sorted = df_val_ave.sort_values(by = "True", ascending = True)

                print("Changing pred column to strings")
                df_val_ave_sorted['Predicted'] = df_val_ave_sorted['Predicted'].astype(str)
                df_val_ave_sorted['True'] = df_val_ave_sorted['True'].astype(str)


                print("Splicing the brackets")
                df_val_ave_sorted['Predicted'] = df_val_ave_sorted['Predicted'].map(lambda x: x.lstrip('[]').rstrip(']'))

                print("Changing columns to floats")
                df_val_ave_sorted['Predicted'] = df_val_ave_sorted['Predicted'].astype(float)
                df_val_ave_sorted['True'] = df_val_ave_sorted['True'].astype(float)

                print("averaging sorted df")
                df_val_averaged = df_val_ave_sorted.groupby(['True']).mean().reset_index()

                print("separating true and pred values")
                tval_dfvalues = df_val_averaged.iloc[:, 0].values
                pval_dfvalues = df_val_averaged.iloc[:, 1].values
                
                # print("Round the pred values if needed")
                tval_values.append(tval_dfvalues)
                pval_values.append(pval_dfvalues)     #np.round(ptrain_values, 3))
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        elif file_csv.startswith('test'):  # Ensure that the file is a CSV
            print(f"Loading csv file: {file_csv}")
            
            try:
                df_test_ave = pd.read_csv(file_path)

                print("sorting df")
                df_test_ave_sorted = df_test_ave.sort_values(by = "True", ascending = True)

                print("Changing pred column to strings")
                df_test_ave_sorted['Predicted'] = df_test_ave_sorted['Predicted'].astype(str)
                df_test_ave_sorted['True'] = df_test_ave_sorted['True'].astype(str)


                print("Splicing the brackets")
                df_test_ave_sorted['Predicted'] = df_test_ave_sorted['Predicted'].map(lambda x: x.lstrip('[]').rstrip(']'))

                print("Changing columns to floats")
                df_test_ave_sorted['Predicted'] = df_test_ave_sorted['Predicted'].astype(float)
                df_test_ave_sorted['True'] = df_test_ave_sorted['True'].astype(float)

                print("averaging sorted df")
                df_test_averaged = df_test_ave_sorted.groupby(['True']).mean().reset_index()

                print("separating true and pred values")
                ttest_dfvalues = df_test_averaged.iloc[:, 0].values
                ptest_dfvalues = df_test_averaged.iloc[:, 1].values

                ttest_values.append(ttest_dfvalues)
                ptest_values.append(ptest_dfvalues)     #np.round(ptrain_values, 3))
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    print(np.shape(ttrain_values), len(ptrain_values[0]), len(tval_values[0]), len(pval_values[0]), np.shape(ttest_values), np.shape(ptest_values))
    
    return ttrain_values, ptrain_values, tval_values, pval_values, ttest_values, ptest_values


# Create a histogram plotting function detailing the frequency of true vs predicted ellipticity values
def histogram(run_data):
    train_true_values, train_pred_values, val_true_values, val_pred_values, test_true_values, test_pred_values = run_data[0]
    run_name = run_data[1]

    # Plot a histogram of true training values vs predicted training values
    print("Plotting training data.")
    plt.figure(1, figsize=(4.5, 3.75))
    plt.hist(train_true_values, bins=20, color='blue', alpha=0.5, label='True Labels $(y_{{true}})$')
    plt.hist(train_pred_values, bins=20, color='red', alpha=0.5, label='Predicted Labels $(y_{{pred}})$')
    plt.title(f"{run_name} Training Data")
    plt.xlabel("Ellipticity Values")
    plt.ylabel("Frequency of Values")
    plt.legend(frameon=False, fontsize='small', loc='upper right', numpoints=1)
    plt.tight_layout()  # Adjust the layout to ensure labels fit
    # plt.savefig(f"{run_name}_training_hist.png", dpi = 320)
    plt.show()
    print("Finished plotting training data.")

    # Plot a histogram of true validation values vs predicted validation values
    print("Plotting validation data.")
    plt.figure(2, figsize=(4.5, 3.75))
    plt.hist(val_true_values, bins=20, color='blue', alpha=0.5, label='True Labels $(y_{{true}})$')
    plt.hist(val_pred_values, bins=20, color='red', alpha=0.5, label='Predicted Labels $(y_{{pred}})$')
    plt.title(f"{run_name} Validation Data")
    plt.xlabel("Ellipticity Values")
    plt.ylabel("Frequency of Values")
    plt.legend(frameon=False, fontsize='small', loc='upper right', numpoints=1)
    plt.tight_layout()  # Adjust the layout to ensure labels fit
    # plt.savefig(f"{run_name}_validation_hist.png", dpi = 320)
    plt.show()
    print("Finished plotting validation data.")

    # Plot a histogram of true test values vs predicted test values
    print("Plotting testing data.")
    plt.figure(3, figsize=(4.5, 3.75))
    plt.hist(test_true_values, bins=20, color='blue', alpha=0.5, label='True Labels $(y_{{true}})$')
    plt.hist(test_pred_values, bins=20, color='red', alpha=0.5, label='Predicted Labels $(y_{{pred}})$')
    plt.title(f"{run_name} Testing Data")
    plt.xlabel("Ellipticity Values")
    plt.ylabel("Frequency of Values")
    plt.legend(frameon=False, fontsize='small', loc='upper right', numpoints=1)
    plt.savefig(f"{run_name}_testing_hist.png", dpi = 320)
    plt.tight_layout()
    plt.show()
    print("Finished plotting testing data.")


# Create a scatterplot plotting function that plots the averaged training and validation data for a single run. It includes testing data as well for comparison
def scatterplot(run_data):
    ttrain_values, ptrain_values, tval_values, pval_values, ttest_values, ptest_values = run_data[0]
    run_name = run_data[1]

    print(len(ttrain_values), len(tval_values), len(ttest_values))

    # Plot a scatterplot of true training values vs predicted validation values
    print("Plotting average data per run data.")
    plt.figure(figsize=(4.5, 3.75))  # Removed figure number to avoid conflict
    plt.scatter(ttrain_values, ptrain_values, c='red', marker='*', label='Training')
    plt.scatter(tval_values, pval_values, c='blue', marker='x', label='Validation')
    plt.scatter(ttest_values, ptest_values, c='green', marker='o', label='Testing')
    plt.plot((0,0.4), (0, 0.4), linestyle = "--", c = 'k', label = '1 to 1 fit')
    plt.title(f"{run_name} $y_{{true}}$ vs $y_{{pred}}$ Values")
    plt.xlabel("True Ellipticity Values")
    plt.ylabel("Predicted Ellipticity Values")
    plt.grid(True)
    plt.legend()

    # Save the plot before showing it
    plt.savefig(f"{run_name}_scatterplot.png", dpi = 320)
    plt.show()

    print("Finished plotting average data per run.")


# Plot the averaged true and predicted ellipticity values for two runs (run1 and run2), with distinct colors and markers for training, validation, and testing datasets of each run
def all_runs_colors_scatterplot(run1_data, run2_data):
    print("all_runs_colors_scatterplot")
    ttrain1_values, ptrain1_values, tval1_values, pval1_values, ttest1_values, ptest1_values = run1_data[0]
    run_name1 = run1_data[1]
    
    ttrain2_values, ptrain2_values, tval2_values, pval2_values, ttest2_values, ptest2_values = run2_data[0]
    run_name2 = run2_data[1]

    # Plot a scatterplot of true training values vs predicted validation values
    # Plotting the true labels on x axis and predicted on y
    print("Plotting average data per run data.")
    plt.figure(figsize=(10, 5))
    plt.scatter(ttrain1_values, ptrain1_values, c='red', marker='*', label = 'Training1')
    plt.scatter(tval1_values, pval1_values, c='blue', marker='x', label = 'Validation1')
    plt.scatter(ttest1_values, ptest1_values, c='green', marker='o', label = 'Testing1')
    plt.scatter(ttrain2_values, ptrain2_values, c='c', marker='*', label = 'Training2')
    plt.scatter(tval2_values, pval2_values, c='k', marker='x', label = 'Validation2')
    plt.scatter(ttest2_values, ptest2_values, c='m', marker='o', label = 'Testing2')
    plt.title(f"{run_name1} + {run_name2} Ave True vs Pred e Values")
    plt.xlabel("True Ellipticity Values")
    plt.ylabel("Predicted Ellipticity Values")
    # Enable grid
    plt.grid(True)
    # Add a legend
    plt.legend()
    # Show the plot
    plt.show()
    print("Finished plotting average data per run.")

    
# Plot the true and predicted values for multiple runs (not limited to just two). It uses a single color for all training, validation, and testing datasets of a specific run but distinguishes them with different markers
def all_runs_scatterplot(all_runs):
    print("Plotting validation data.")
    plt.figure(figsize=(10, 5))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for run, color in zip(all_runs, colors):
        train_values_true, train_values_pred, val_values_true, val_values_pred, test_values_true, test_values_pred = run[0]
        name_run = run[1]

        # Use the same color for all scatter plots of the current run
        plt.scatter(train_values_true, train_values_pred, c=color, marker='*', label=f'{name_run} Train')
        plt.scatter(val_values_true, val_values_pred, c=color, marker='o', label=f'{name_run} Validation')
        plt.scatter(test_values_true, test_values_pred, c=color, marker='x', label=f'{name_run} Test')

    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of True vs Predicted Values for All Runs')
    plt.legend()
    plt.grid(True)
    plt.show()


# Create a violin plot to visually compare the distributions of true and predicted ellipticity values across training, validation, and testing datasets for a specified run
def violin_plot(run_data):
    train_true_values, train_pred_values, val_true_values, val_pred_values, test_true_values, test_pred_values = run_data[0]
    run_name = run_data[1]

    # Ensure that all arrays are 1-dimensional
    train_true_values = np.ravel(train_true_values)
    train_pred_values = np.ravel(train_pred_values)
    val_true_values = np.ravel(val_true_values)
    val_pred_values = np.ravel(val_pred_values)
    test_true_values = np.ravel(test_true_values)
    test_pred_values = np.ravel(test_pred_values)

    # Combine true and predicted values into a DataFrame for plotting
    data = {
        'True Values': np.concatenate([train_true_values, val_true_values, test_true_values]),
        'Predicted Values': np.concatenate([train_pred_values, val_pred_values, test_pred_values]),
        'Dataset': ['Train'] * len(train_true_values) + ['Validation'] * len(val_true_values) + ['Test'] * len(test_true_values)
    }
    
    df = pd.DataFrame(data)

    # Create the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Dataset', y='True Values', data=df, inner='quartile', color='lightblue', label='True Values')
    sns.violinplot(x='Dataset', y='Predicted Values', data=df, inner='quartile', color='salmon', label='Predicted Values')
    plt.title(f"{run_name} Violin Plot of True vs Predicted Ellipticity Values")
    plt.xlabel("Dataset")
    plt.ylabel("Ellipticity Values")
    plt.grid(True)
    plt.legend()
    plt.show()


# Plot Histograms
best_model = (process_csvs(best_run_folder_path), "Best Model")
second_best_model = (process_csvs(second_best_folder_path), "2nd Best Model")

best_plots = histogram(best_model)
second_best_plots = histogram(second_best_model)

# Get averages per run
best_averages = (average_preds(best_run_folder_path), "Best Model")
second_best_averages = (average_preds(second_best_folder_path), "2nd Best Model")
# pdb.set_trace()
both_runs = [best_averages, second_best_averages]

# Plot individual scatterplots of averages by run
best_scatterplot = scatterplot(best_averages)
second_best_scatterplot = scatterplot(second_best_averages)

# Plot overlapped scatterplot of averages
both_runs_plots = all_runs_scatterplot(both_runs)
both_runs_colors_plots = all_runs_colors_scatterplot(best_averages, second_best_averages)

# Create violin plots of the runs
violin_plot(best_averages, "Best Model")
violin_plot(second_best_averages, "2nd Best Model")