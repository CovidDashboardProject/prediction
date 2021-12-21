import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from datetime import date
import datetime
import gspread

from oauth2client import client
from oauth2client.service_account import ServiceAccountCredentials

# from cessi.download_data import get_data
import pandas as pd
import requests




def connect_to_db(credentials):

    scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials,scope)
    client = gspread.authorize(creds)

    return client
  
def connect():

  credentials = 'creds.json'
  client = connect_to_db(credentials)

  return client

def get_data_sheet(client,sheet_name):

    return client.open(sheet_name).sheet1


def get_pred_sheet():

    sheet_name = 'Predictions'

    client = connect()

    sheet = get_data_sheet(client,sheet_name)

    return sheet


def get_cessi_data(client):

  cessi_data_sheet = client.open('cessi_data_file').sheet1
  cessi_data = pd.DataFrame(cessi_data_sheet.get_all_records())

  cessi_data['Date'] = pd.to_datetime(cessi_data['Date'],format = '%Y-%m-%d')
  cessi_data = cessi_data.iloc[12:] # to match both of the data ( some of the data in cessi is missing (24-4-2020 and 25-4-2020))
  
  return cessi_data

def get_mobility_data(client):

  mobility_sheet = client.open('MH_mobility').worksheets()

  mobility_data_sheet_2020 = mobility_sheet[1]
  mobility_data_sheet_2021 = mobility_sheet[0]

  mobility_data_2020 = pd.DataFrame(mobility_data_sheet_2020.get_all_records())
  mobility_data_2021 = pd.DataFrame(mobility_data_sheet_2021.get_all_records())

  mobility_data = pd.concat([mobility_data_2020,mobility_data_2021],axis=0)
  mobility_data = mobility_data[mobility_data['sub_region_2'] == 'Pune']
  mobility_data = mobility_data.drop(columns = ['sub_region_1','sub_region_2'])
  
  mobility_data = mobility_data.iloc[71:]
  mobility_data['date'] = pd.to_datetime(mobility_data['date'],format = '%Y-%m-%d')

  return mobility_data


def get_data():

  client = connect()

  cessi_data = get_cessi_data(client)
  mobility_data = get_mobility_data(client)
  
  columns = list(cessi_data.columns) + list(mobility_data.columns)
  final_data = pd.DataFrame(columns = columns)

  for ( _ , row1 ) , (_ , row2) in zip(cessi_data.iterrows(),mobility_data.iterrows()):
    row = list(row1) + list(row2)
    final_data.loc[len(final_data.index)] = row

  final_data.drop(columns = ['date'],inplace = True)

  return final_data

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


def compile_and_fit(model, window, patience=2,MAX_EPOCHS = 50):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)

    self.dense = tf.keras.layers.Dense(num_features)

def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)
  

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup


def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the LSTM state.
  prediction, state = self.warmup(inputs)

  # Insert the first prediction.
  predictions.append(prediction)

  # Run the rest of the prediction steps.
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output.
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call
