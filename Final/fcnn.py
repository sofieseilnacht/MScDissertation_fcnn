import numpy as np
import pdb
import pandas as pd
import os
import pickle
import psutil
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, regularizers


print("Starting script...")


# Define paths to pickle folder and memory map files
pickle_file_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/final_pickles'
real_data_memmap_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/dat_files/real_data_run9.dat'
imag_data_memmap_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/dat_files/imag_data_run9.dat'
ellipticity_memmap_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/dat_files/ellipticity_run9.dat'


# Print memory to check RAM usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB")


# Open pickle files and load in the data 
def load_data_from_pickle(pickle_file_path):
    print(f"Loading pickle file: {pickle_file_path}")
    with open(pickle_file_path, 'rb') as f:
        return pickle.load(f)


# Normalize the data
def normalize_data(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        # Avoid division by zero by returning zeros if min == max
        return np.zeros_like(data)  
    return (data - data_min) / (data_max - data_min)


# Process the pickle file data (normalize and separate the input and output data, then save them to disc using memory mapping) 
def prepare_data_for_tf_dataset(input_output_list, real_data, imag_data, ellipticity, start_index):
    print("Begin preparing data from pickle files.")
    index = start_index
    for outer_dict in input_output_list:
        print("Processing outer_dict...")
        for nested_dict in outer_dict.values():
            # Fourier pixel data is the input data and ellipticity is the output labels
            fft_data = nested_dict.get('fourier_pixel_data', None)
            ellipticity_value = nested_dict.get('ellipticity', None)
            if fft_data is not None and ellipticity_value is not None:
                # Separate data into real and complex components
                real = np.real(fft_data)
                imag = np.imag(fft_data)

                # Normalize data
                real = normalize_data(real)
                imag = normalize_data(imag)

                # Store combined data
                real_data[index] = real
                imag_data[index] = imag

                if ellipticity_value is not None:
                    ellipticity[index] = ellipticity_value
                index += 1
            else:
                print("data was none")

    # Flushing data to memory mapped files on computer disc
    print("Flushing memory-mapped files...")
    real_data.flush()
    imag_data.flush()
    ellipticity.flush()

    return index


# Load and process the pickle file data
def pickle_data_generator(pickle_folder_path, batch_files, file_batch_size):
    def load_and_process_batch(batch_files_to_process, start_index):
        input_output_list = []
        for batch_file in batch_files_to_process:
            print(f"Loading batch file.")
            try:
                batch_data = load_data_from_pickle(os.path.join(pickle_folder_path, batch_file))
                fft_batch = batch_data[0]
                input_output_list.append(fft_batch)
                print(f"Loaded pickle file: {batch_file}")
            except Exception as e:
                print(f"Error loading pickle file {batch_file}: {e}")
        
        if input_output_list:
            print("Preparing data for TensorFlow Dataset.")
            return prepare_data_for_tf_dataset(input_output_list, real_data, imag_data, ellipticity, start_index)

        return start_index

    # Define parameters
    # Total number of samples calculation
    print("Initializing the memory mapped files.")
    num_samples_per_batch = 202  # Based on the number of entries in outer_dict
    total_batches = len(batch_files)
    num_data_samples = num_samples_per_batch * total_batches
    print(f"Total number of data samples: {num_data_samples}")    

    data_shape = (199, 199)
    ellipticity_shape = (num_data_samples,)
    print(f"Data Shape: {data_shape}")
    print(f"Ellipticity Shape: {ellipticity_shape}")

    # It is imperative to define the expected shape of the data for the memory maps to know exactly how much data to store
    real_data = np.memmap(real_data_memmap_path, dtype='float32', mode='w+', shape=(num_data_samples, *data_shape))
    imag_data = np.memmap(imag_data_memmap_path, dtype='float32', mode='w+', shape=(num_data_samples, *data_shape))
    ellipticity = np.memmap(ellipticity_memmap_path, dtype='float32', mode='w+', shape=ellipticity_shape)

    start_index = 0

    # Batch the data to lessen memory usage
    for batch_count in range(0, len(batch_files), file_batch_size):
        print(f"Processing batch {batch_count // file_batch_size + 1}/{(len(batch_files) + file_batch_size - 1) // file_batch_size}...")
        batch_files_to_process = batch_files[batch_count:batch_count + file_batch_size]
        start_index = load_and_process_batch(batch_files_to_process, start_index)

    return real_data, imag_data, ellipticity, num_data_samples


# Use data generators to process data on the fly utilizing the TF pipeline
def data_generator(indices, batch_size, data_memmap, label_memmap):
    print("Generating data from generators.")
    num_indices = len(indices)

    # Matching real and imaginary portions of the data with their corresponding labels
    for start in range(0, num_indices, batch_size):
        end = min(start + batch_size, num_indices)
        batch_indices = indices[start:end]
        real_batch = data_memmap['real'][batch_indices]
        imag_batch = data_memmap['imag'][batch_indices]
        ellip_batch = label_memmap[batch_indices]

        # Add channel dimension to real and imaginary batches
        real_batch = np.expand_dims(real_batch, axis=-1)  # Shape: (batch_size, 199, 199, 1)
        imag_batch = np.expand_dims(imag_batch, axis=-1)  # Shape: (batch_size, 199, 199, 1)

        # Concatenate real and imaginary batches along the last axis
        combined_batch = np.concatenate([real_batch, imag_batch], axis=-1)  # Shape: (batch_size, 199, 199, 2)

        yield (
            combined_batch,
            ellip_batch
        )


# create and prepare the data set for our model using the TF pipeline and the data generator function
def create_tf_dataset(indices, batch_size, data_memmap, label_memmap):
    print("Creating the TF datasets.")
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(indices, batch_size, data_memmap, label_memmap),
        output_signature=(
            tf.TensorSpec(shape=(None, 199, 199, 2), dtype=tf.float32),  # Updated shape
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    # Ensure the dataset repeats if necessary
    # dataset = dataset.repeat()
    return dataset


# Build the model
def create_model(input_shape):
    print("Creating model with input shape:", input_shape)
    
    input_data = tf.keras.Input(shape=input_shape, name='input_data')

    # Split the input into real and imaginary parts
    input_real = input_data[..., 0:1]  # Shape: (batch_size, 199, 199, 1)
    input_imag = input_data[..., 1:2]  # Shape: (batch_size, 199, 199, 1)

    # Process real data
    x_real = Conv2D(32, (3, 3), activation='relu', padding='same')(input_real)
    x_real = MaxPooling2D((2, 2))(x_real)
    x_real = Conv2D(64, (3, 3), activation='relu', padding='same')(x_real)
    x_real = MaxPooling2D((2, 2))(x_real)
    x_real = Conv2D(128, (3, 3), activation='relu', padding='same')(x_real)
    x_real = MaxPooling2D((2, 2))(x_real)

    # Process imaginary data
    x_imag = Conv2D(32, (3, 3), activation='relu', padding='same')(input_imag)
    x_imag = MaxPooling2D((2, 2))(x_imag)
    x_imag = Conv2D(64, (3, 3), activation='relu', padding='same')(x_imag)
    x_imag = MaxPooling2D((2, 2))(x_imag)
    x_imag = Conv2D(128, (3, 3), activation='relu', padding='same')(x_imag)
    x_imag = MaxPooling2D((2, 2))(x_imag)

    # Concatenate the processed real and imaginary data
    x = Concatenate()([x_real, x_imag])
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input_data, outputs=output)
    # model.summary()
    return model


# Define batch files and batch size
batch_files = [f for f in os.listdir(pickle_file_path) if f.startswith(('batch_')) and f.endswith('.pkl')]
file_batch_size = 3
chunk_size = 100  # Adjust based on available memory
batch_size = 64  # Adjust based on available memory


# Create data generator and process all batches
real_data, imag_data, ellipticity, num_data_samples = pickle_data_generator(pickle_file_path, batch_files, file_batch_size)


print("Data preparation complete.")


# Create indices and shuffle them
print("Creating indices to shuffle data.")
indices = np.arange(num_data_samples)
np.random.shuffle(indices)


# Split indices into training, validation, and test sets
print("Splitting indices to training, validation, and test sets.")
train_split = int(0.67 * num_data_samples)
val_split = int(0.25 * train_split)

train_indices = indices[:train_split]
val_indices = indices[train_split:train_split + val_split]
test_indices = indices[train_split + val_split:]
sum = len(train_indices)+len(val_indices)+len(test_indices)
print(f"total indices (train, test, val): {sum}")


# Create TensorFlow datasets
print("Creating TensorFlow dataset.")
data_memmap = {'real': real_data, 'imag': imag_data}
label_memmap = ellipticity


# pdb.set_trace()


print("Splitting memmapped into training, testing, and validation sets according to indices.")
train_dataset = create_tf_dataset(train_indices, batch_size, data_memmap, label_memmap)
val_dataset = create_tf_dataset(val_indices, batch_size, data_memmap, label_memmap)
test_dataset = create_tf_dataset(test_indices, batch_size, data_memmap, label_memmap)


# Model creation and compilation
print("Creating and compiling model.")
input_shape = (199, 199, 2)
model = create_model(input_shape)


def difference_metric(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred), axis=-1)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def r2_score(y_true, y_pred):
    ss_total = K.sum(K.square(y_true - K.mean(y_true)))
    ss_residual = K.sum(K.square(y_true - y_pred))
    return 1 - (ss_residual / (ss_total + K.epsilon()))


# Ensure no conflicts in metric names
metrics = {
    'difference_metric': difference_metric,
    'mse': 'mse',
    'mae': 'mae',
    'rmse': rmse,
    'mape': 'mape',
    'r2_score': r2_score    
}


# Write the predictions to a csv file as a Pandas df
def log_predictions(model, dataset, output_csv_path):
    true_values = []
    pred_values = []

    # Iterate over dataset
    for x_batch, y_true in dataset:
        y_pred = model.predict(x_batch)
        
        # Convert tensors to numpy arrays if needed
        true_values.extend(y_true.numpy())  # Convert tensors to numpy arrays and extend
        pred_values.extend(y_pred)  # No need to call .numpy() on y_pred if it is already a numpy array

    # Create DataFrame
    results_df = pd.DataFrame({'True': true_values, 'Predicted': pred_values})
    
    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


# Define a custom callback to log predictions to CSV files at the end of each epoch
class prediction_logger_callback(tf.keras.callbacks.Callback):
    def __init__(self, training_data=None, validation_data=None, batch_size=1, output_csv_path = 'predictions_epoch{epoch}.csv'):
                 # OR output_csv_path='predictions_epoch_{epoch}_{type}.csv'):
        super(prediction_logger_callback, self).__init__()
        self.training_data = training_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.output_csv_path = output_csv_path

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}: Logging Predictions")

        # Generate paths for train and validation predictions
        train_csv_path = self.output_csv_path.format(epoch=epoch+1, type='train')
        val_csv_path = self.output_csv_path.format(epoch=epoch+1, type='val')

        # Log training predictions
        if self.training_data is not None:
            log_predictions(self.model, self.training_data, train_csv_path)
        
        # Log validation predictions
        if self.validation_data is not None:
            log_predictions(self.model, self.validation_data, val_csv_path)


# Instantiate the callback
prediction_logger = prediction_logger_callback(
    training_data=train_dataset, 
    validation_data=val_dataset, 
    batch_size=batch_size, 
    output_csv_path='/Users/sofie/Desktop/Projects/Dissertation_Program/predictions_run9/preds_run9_epoch{epoch}_{type}.csv'
)

# Instantiate earlystopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',       # Metric to monitor
    patience=5,                # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
)


# Define the optimizer and compile the model
optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer='adam',
              loss='mae',
              metrics=[difference_metric, 'mse', 'mae', rmse, 'mape', r2_score])


# Model training
print("Fitting/training the model.")
training_results = model.fit(train_dataset, 
                             validation_data=val_dataset, 
                             steps_per_epoch=len(train_indices) // batch_size, 
                             epochs=30, 
                             callbacks= [prediction_logger, early_stopping_callback]) # [checkpoint_callback, prediction_logger])


# Extract metrics and loss from the training history object and log it in csv files
training_results_dict = training_results.history
metric_names = list(training_results.history.keys())
history_df = pd.DataFrame(training_results.history)
training_results_csv_file_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/training_history_run9.csv'
history_df.to_csv(training_results_csv_file_path, index=False)
print(f"Training history saved to {training_results_csv_file_path}")
for metric_name in metric_names:
    print(f"{metric_name}: {training_results.history[metric_name]}")


# Assuming you have a test dataset
print("Logging predictions for test data.")
log_predictions(model, test_dataset, 'test_predictions_run9.csv')


# Evaluate the model
print("Evaluating the model.")
evaluation_results = model.evaluate(test_dataset, batch_size=batch_size, verbose=1, return_dict=True)


# Extract metric names and results from the evaluation dictionary
metric_names = list(evaluation_results.keys())
metric_results = [evaluation_results[name] for name in metric_names]


# Create a DataFrame from evaluation results
evaluation_data = { 'Metric': metric_names, 'Result': metric_results }
evaluation_df = pd.DataFrame(evaluation_data)
evaluation_csv_file_path = '/Users/sofie/Desktop/Projects/Dissertation_Program/evaluation_results_run9.csv'
evaluation_df.to_csv(evaluation_csv_file_path, index=False)
print(f"Evaluation results saved to {evaluation_csv_file_path}")


# Print results
for name, result in zip(metric_names, metric_results):
    print(f"{name}: {result}")


# Check if the number of results matches the number of metric names
if len(metric_results) != len(metric_names):
    print("Error: The number of evaluation results does not match the number of metric names.")
    print(f"Evaluation results: {metric_results}")
    print(f"Metric names: {metric_names}")
