# model is the class that has the compilation functions on layers.
# It is a stepping stone to building sophisticated deep learning models with flexible architectures in TensorFlow.
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd


train_df = pd.read_csv('../data/train.csv', index_col=0)
val_df = pd.read_csv('../data/val.csv', index_col=0)
test_df = pd.read_csv('../data/test.csv', index_col=0)


# Basic model copies the last value once
class Baseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    # it overrides the function call in class Model (that is shown below)
    # in each case we copy the input as prediction
    def call(self, inputs):
        # if no specified target we take all columns
        if self.label_index is None:
            return inputs

        # else we operate the selection if there is a list of target
        elif isinstance(self.label_index, list):
            tensors = []
            for index in self.label_index:
                result = inputs[:, :, index]
                # new axis adds a dimension cf below for example
                result = result[:, :, tf.newaxis]
                tensors.append(result)
            return tf.concat(tensors, axis=-1)

        # finally we consider the case of a single label
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


# in this version we adapt by changing the call method in order to predict the 24 next values as the last one
class MultiStepLastBaseline(Baseline):
    def __init__(self, label_index=None):
        super().__init__(label_index=label_index)

    def call(self, inputs):
        if self.label_index is None:
            return tf.tile(inputs[:, -1:, :], [1, 24, 1])
        return tf.tile(inputs[:, -1:, self.label_index:], [1, 24, 1])


# in this version we adapt by changing the call method in order to
# predict the next values as the copy of the previous ones
class RepeatBaseline(Baseline):
    def __init__(self, label_index=None):
        super().__init__(label_index=label_index)

    def call(self, inputs):
        return inputs[:, :, self.label_index:]


"""
        def call(self, inputs, training=None, mask=None):
        Calls the model on new inputs.

        In this case `call` just reapplies
        all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs).

        Note: This method should not be called directly. It is only meant to be
        overridden when subclassing `tf.keras.Model`.
        To call a model on an input, always use the `__call__` method,
        i.e. `model(inputs)`, which relies on the underlying `call` method.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            training: Boolean or boolean scalar tensor, indicating whether to run
              the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
"""

"""
import tensorflow as tf

# Create a 1D tensor
a = tf.constant([1, 2, 3])

# Reshape using tf.newaxis
# Adds a new dimension at the start
b = a[tf.newaxis, :]

# Adds a new dimension at the end
c = a[:, tf.newaxis]

print("Original shape:", a.shape)
print("New shape (start):", b.shape)
print("New shape (end):", c.shape)

outputs---> 
Original shape: (3,)
New shape (start): (1, 3)
New shape (end): (3, 1)

"""


# for each baseline we will have the size of input and output that can be
# completely flexible, and you can pass labels of columns instead of index
class MetaBaseline(Model):
    def __init__(self, train_df_input=train_df, label_columns=None, label_index=None,
                 output_prediction_window_size=1, input_window_size=1):
        super().__init__()
        self.input_window_size = input_window_size
        self.train_df = train_df_input
        self.output_prediction_window_size = output_prediction_window_size
        self.label_columns = label_columns
        if label_index is None and label_columns is not None:
            self.label_index = [self.train_df.columns.get_loc(col) for col in label_columns]
        else:
            self.label_index = label_index


# so for each case we have to define call that has 3 cases :
# a single label in input/output, multiple (select them), none (select all)

# this version copies in output the value that is the average of the input
class AverageBaseline(MetaBaseline):
    def __init__(self, train_df_input=train_df, label_columns=None, label_index=None,
                 output_prediction_window_size=1, input_window_size=1):
        super().__init__(train_df_input, label_columns, label_index, output_prediction_window_size, input_window_size)

    def call(self, inputs):
        if self.label_index is None:
            # Calculate the average of the last 'input_window_size' values
            result = tf.reduce_mean(inputs[:, -self.input_window_size:, :], axis=1, keepdims=True)

        elif isinstance(self.label_index, list):
            tensors = []
            for index in self.label_index:
                # Calculate the average of the last 'input_window_size' values for each specified column
                avg_last_values = tf.reduce_mean(inputs[:, -self.input_window_size:, index], axis=1, keepdims=True)
                # Add a new axis to match the required shape
                avg_last_values = avg_last_values[:, :, tf.newaxis]
                tensors.append(avg_last_values)
            # Concatenate all tensors along the last axis
            result = tf.concat(tensors, axis=-1)

        else:
            # Calculate the average of the last 'input_window_size' values for specified features
            result = tf.reduce_mean(inputs[:, -self.input_window_size:, self.label_index:], axis=1, keepdims=True)

        # Tile the average value 'output_prediction_window_size' times along the time step axis
        return tf.tile(result, [1, self.output_prediction_window_size, 1])


# this version copies in output the value that is the last value of the input
class LastValueBaseline(MetaBaseline):
    def __init__(self, train_df_input=train_df, label_columns=None, label_index=None,
                 output_prediction_window_size=1, input_window_size=1):
        super().__init__(train_df_input, label_columns, label_index, output_prediction_window_size, input_window_size)

    def call(self, inputs):
        if self.label_index is None:
            result = inputs[:, -1:, :]

        elif isinstance(self.label_index, list):
            # Create a list to hold the last value tensors
            tensors = []
            for index in self.label_index:
                # Copy the last value for each specified column
                last_value = inputs[:, -1, index]  # Shape: [batch_size]
                last_value = tf.reshape(last_value, [-1, 1, 1])  # Reshape to [batch_size, 1, 1]
                tensors.append(last_value)
            # Concatenate all tensors along the last axis
            result = tf.concat(tensors, axis=-1)
            # Replicate the result across the time step dimension
        else:
            result = inputs[:, -1:, self.label_index:]

        return tf.tile(result, [1, self.output_prediction_window_size, 1])


# this version copies in output the input, if the size input-output differ
# handle_size_mismatch will take care of that and cycle or truncate the input to fit the output size
class CopyBaseline(MetaBaseline):
    def __init__(self, train_df_input=train_df, label_columns=None, label_index=None,
                 output_prediction_window_size=1, input_window_size=1):
        super().__init__(train_df_input, label_columns, label_index, output_prediction_window_size, input_window_size)

    # Define a function to handle different input - output size scenarii
    @staticmethod
    def handle_size_mismatch(input_tensor, input_w, output_w):
        if input_w == output_w:
            # If input size equals output size, copy inputs directly
            return input_tensor

        elif input_w > output_w:
            # If input size is greater, truncate inputs to match output size
            return input_tensor[:, :output_w, :]

        else:
            # If input size is smaller, cycle over inputs until output size is reached
            num_repeats = tf.cast(tf.math.ceil(output_w / input_w), tf.int32)
            repeated_inputs = tf.tile(input_tensor, [1, num_repeats, 1])
            return repeated_inputs[:, :output_w, :]

    def call(self, inputs):
        input_shape = tf.shape(inputs)  # Get the dynamic shape of the input
        batch_size, input_width, num_features = input_shape[0], input_shape[1], input_shape[2]

        # Determine the size of the output based on the replication count
        output_width = self.output_prediction_window_size

        # Handling different data formats
        if self.label_index is None:
            # Process the entire input tensor
            return self.handle_size_mismatch(inputs, input_width, output_width)

        elif isinstance(self.label_index, list):
            # Process each column specified in label_index
            processed_tensors = []
            for index in self.label_index:
                column_tensor = inputs[:, :, index:index+1]  # Extract the column tensor
                processed_tensor = self.handle_size_mismatch(column_tensor, input_width, output_width)
                processed_tensors.append(processed_tensor)
            return tf.concat(processed_tensors, axis=-1)

        else:
            # Process a single specified feature
            feature_tensor = inputs[:, :, self.label_index:self.label_index+1]
            return self.handle_size_mismatch(feature_tensor, input_width, output_width)
