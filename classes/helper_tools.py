


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import warnings
from sklearn.preprocessing import MinMaxScaler
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell


def print_perf(_performance, _val_performance):
    mae_val = [v[1] for v in _val_performance.values()]
    mae_test = [v[1] for v in _performance.values()]

    x = np.arange(len(_performance))

    for v in _val_performance.values():
        print(v)

    fig, ax = plt.subplots()
    ax.bar(x - 0.15, mae_val, width=0.25, color='black', edgecolor='black', label='Validation')
    ax.bar(x + 0.15, mae_test, width=0.25, color='white', edgecolor='black', hatch='/', label='Test')
    ax.set_ylabel('Mean absolute error')
    ax.set_xlabel('Models')

    for index, value in enumerate(mae_val):
        plt.text(x=index - 0.15, y=value + 0.0025, s=str(round(value, 3)), ha='center')

    for index, value in enumerate(mae_test):
        plt.text(x=index + 0.15, y=value + 0.0025, s=str(round(value, 3)), ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(_performance.keys(), rotation=45, ha='right')

    plt.ylim(0, 0.5)
    plt.xticks(ticks=x, labels=_performance.keys())
    plt.legend(loc='best')
    plt.tight_layout()


def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history