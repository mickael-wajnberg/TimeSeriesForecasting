import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from baseline import MetaBaseline


class testClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"


train_df = pd.read_csv('../data/train.csv', index_col=0)
val_df = pd.read_csv('../data/val.csv', index_col=0)
test_df = pd.read_csv('../data/test.csv', index_col=0)


class DataWindow:
    def __init__(self, input_width, label_width, shift,
                 train_df_input=train_df, val_df_input=val_df, test_df_input=test_df,
                 label_columns=None):
        # the datasets
        self._sample_batch = None
        self.train_df = train_df_input
        self.val_df = val_df_input
        self.test_df = test_df_input

        # label of the columns we want to predict, if not empty, create a dictionary
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # in a window we have
        # https://www.tensorflow.org/static/tutorials/structured_data/images/raw_window_24h.png
        # - the input (how much is used to train)
        self.input_width = input_width
        # - the number of labels  (how much is predicted)
        self.label_width = label_width
        # - the distance in the time which you have to predict
        self.shift = shift

        self.total_window_size = self.input_width + self.shift

        # start and end point of input part of a window
        self.input_slice = slice(0, input_width)

        # add labels per window for plotting
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # position to start prediction
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    # the next two functions are the most important, that is where inputs for baseline is created !
    # this function takes the dataset (wrt the selected column) and splits it into input/labels in order to have
    # prediction made with inputs and compare it to labels(real values)
    # this is used in make_dataset
    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # takes an array and makes slices
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )
        # for each slice we cut it into inputs and label
        ds = ds.map(self.split_to_inputs_labels)
        return ds

    def plot(self, model=None, plot_col='traffic_volume', max_subplots=3):
        inputs, labels = self.sample_batch
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='red', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time (h)')

    # The @property allows the method to be accessed like an attribute rather than as a method
    # with parentheses x.train=x.train()
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    # is used for plotting
    @property
    def sample_batch(self):
        # check if there is something in cache mem
        result = getattr(self, '_sample_batch', None)
        if result is None:
            # if cache empty compute and cache 1
            result = next(iter(self.train))
            self._sample_batch = result
        return result


# here instead of copying model parameter you can simply pass the model and everything
# will be added automatically as long as the model is for a MetaBaseline,
# but you can still use it as previously passing dataset and parameters
# you can also plot all the columns at the same time with plot_upg
class MetaBaselineDataWindow:
    def __init__(self, input_width=None, label_width=None, shift=None,
                 train_df_input=train_df, val_df_input=val_df, test_df_input=test_df,
                 label_columns=None, model=None):
        # the datasets
        self._sample_batch = None
        self.train_df = train_df_input
        self.val_df = val_df_input
        self.test_df = test_df_input
        self.model = model
        self.label_columns = label_columns
        if model is not None:
            self.label_columns = model.label_columns
        # label of the columns we want to predict, if not empty, create a dictionary

        if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # in a window we have
        # https://www.tensorflow.org/static/tutorials/structured_data/images/raw_window_24h.png
        # - the input (how much is used to train)
        self.input_width = input_width
        if self.input_width is None and isinstance(self.model, MetaBaseline):
            self.input_width = model.input_window_size
        elif self.input_width is None and not isinstance(self.model, MetaBaseline):
            raise ValueError("input width is not specified")
        # - the number of labels  (how much is predicted)
        self.label_width = label_width
        if self.label_width is None and isinstance(self.model, MetaBaseline):
            self.label_width = model.output_prediction_window_size
        elif self.label_width is None and not isinstance(self.model, MetaBaseline):
            raise ValueError("label width is not specified")
        # - the distance in the time which you have to predict
        self.shift = shift
        if shift is None and isinstance(self.model, MetaBaseline):
            self.shift = self.label_width
        elif shift is None:
            self.shift = 1

        self.total_window_size = self.input_width + self.shift

        # start and end point of input part of a window
        self.input_slice = slice(0, self.input_width)

        # add labels per window for plotting
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # position to start prediction
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    # this function takes the dataset (wrt the selected column) and splits it into input/labels in order to have
    # prediction made with inputs and compare it to labels(real values)
    # this is used in make_dataset
    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot_upg(self, model=None, plot_col=None, max_subplots=3):
        if model is None and isinstance(self.model, MetaBaseline):
            for col in self.model.label_columns:
                self.plot(self.model, col, max_subplots)
            return

        if plot_col is None:
            for col in train_df.columns:
                self.plot(model, col, max_subplots)
            return
        elif isinstance(plot_col, list):
            for col in plot_col:
                self.plot(model, col, max_subplots)
            return
        return self.plot(model, plot_col, max_subplots)

    def plot(self, model=None, plot_col='traffic_volume', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='red', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time (h)')

    # takes an array and makes slices
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )
        # for each slice we cut it into inputs and label
        ds = ds.map(self.split_to_inputs_labels)
        return ds

    # The @property allows the method to be accessed like an attribute rather than as a method
    # with parentheses x.train=x.train()
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    # is used for plotting
    @property
    def sample_batch(self):
        # check if there is something in cache mem
        result = getattr(self, '_sample_batch', None)
        if result is None:
            # if cache empty compute and cache 1
            result = next(iter(self.train))
            self._sample_batch = result
        return result
