#Cross Validation Run LSTM


#Function to create tensorflow datasets and split datasets into train/val/test

#Function to split numpy arrays X and Y to train and test. Train (80%) and Test (20%)

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras import backend as K

def f1_score(precision, recall):
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#every sample in train/validation/test is a 2d tensor of shape (sequence_length, num_features)

def build_LSTM(sequence_length, num_features):
  inputs = keras.Input(shape=(sequence_length, num_features))
  x=layers.Masking(mask_value=0)(inputs)
  #x = layers.LSTM(16)(inputs)
  x = layers.LSTM(16)(x)
  #The first hidden layer  is a dense/fully connected layer
  x=layers.Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=1))(x)
  #The final output layer has one neuron with sigmoid activation to output the probability of the target class
  outputs=layers.Dense(1, activation="sigmoid")(x)
  #outputs = layers.Dense(1)(x)
  ltsm_model = keras.Model(inputs, outputs)
  return ltsm_model

def learning_rate_schedule():
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)
  return lr_schedule

def split_XY_arrays(X, Y):

  from sklearn.model_selection import train_test_split
  import numpy as np
  """ Create train and test TF dataset from X and Y
    The prefetch overlays the preprocessing and model execution of a training step.
    While the model is executing training step s, the input pipeline is reading the data for step s+1.
    AUTOTUNE automatically tune the number for sample which are prefeteched automatically.

    Keyword arguments:
    X -- numpy array
    Y -- numpy array
    batch_size -- integer
  """
  #AUTOTUNE = tf.data.experimental.AUTOTUNE

  #X = X.astype('float32')
  #Y = Y.astype('float32')

  x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y, test_size = 0.2, random_state=42, stratify=Y, shuffle=True)
  return x_tr, x_ts, y_tr, y_ts


#Function to create Tensorflow dataset without splitting

def create_tf_dataset(X, Y, batch_size):
  """ Create train and test TF dataset from X and Y
    The prefetch overlays the preprocessing and model execution of a training step.
    While the model is executing training step s, the input pipeline is reading the data for step s+1.
    AUTOTUNE automatically tune the number for sample which are prefeteched automatically.

    Keyword arguments:
    X -- numpy array
    Y -- numpy array
    batch_size -- integer
  """
  AUTOTUNE = tf.data.experimental.AUTOTUNE

  X = X.astype('float32')
  Y = Y.astype('float32')

  #x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y, test_size = 0.3, random_state=42, stratify=Y, shuffle=True)

  tf_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
  #print("train length: ", len(tf_dataset))
  # the length function is valid only in eager mode in tensorflow version 2

  # DCB: TypeError: tf.data.Dataset only supports 'len' in eager mode. Use 'tf.data.Dataset.cardinality()' instead.
  # tf_length = len(tf_dataset)
  # tf_length= tf_dataset.cardinality()
  tf_length= tf_dataset.cardinality().numpy()
  tf_dataset = tf_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
  tf_dataset = tf_dataset.batch(batch_size).prefetch(AUTOTUNE)


  return tf_dataset, tf_length

def LSTM_CV(x,y, sequence_length, num_features):

  X_train_array, X_test_array, y_train_array, y_test_array = split_XY_arrays(x, y)
  print("X_train Shape: ", X_train_array.shape)
  print("X_test Shape: ", X_test_array.shape)
  test_tf, test_tf_length = create_tf_dataset(X_test_array, y_test_array, 200)
  print("test length: ", test_tf_length)

  X_training_array, X_val_array, y_training_array, y_val_array = split_XY_arrays(X_train_array, y_train_array)
  print("X_training Shape: ", X_training_array.shape)
  print("X_val Shape: ", X_val_array.shape)

  #Splitting data into multiple folds and training model on each fold every time.
  #from sklearn.model_selection import KFold
  from sklearn.model_selection import StratifiedKFold
  #cv = KFold(n_splits=10, shuffle=True, random_state=42)
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

  # K-fold Cross Validation model evaluation
  fold_no = 1
  #acc_per_fold = [] #save accuracy from each fold
  val_metrics_fold = [] #save metrics from each fold
  test_metrics_fold = []
  lstm_hist = pd.DataFrame()
  lstm_best_score = []

  for train, val in cv.split(X_train_array, y_train_array):

      print('   ')
      print(f'Training for fold {fold_no} ...')

      #Scale data
      #scaler = MinMaxScaler()
      train_X_array = X_train_array[train]
      train_y_array = y_train_array[train]
      val_X_array = X_train_array[val]
      val_y_array = y_train_array[val]
      print("train_X Shape: ", train_X_array.shape)
      print("val_X Shape: ", val_X_array.shape)
      train_tf, train_tf_length = create_tf_dataset(train_X_array, train_y_array, 200)
      val_tf, val_tf_length = create_tf_dataset(val_X_array, val_y_array, 200)
      print("Train  length: ", train_tf_length)
      print("Val  length: ", val_tf_length)

      #test_tf_antiberty = create_tf_dataset(X_test_antiberty_array, y_test_array, 200)
      #scaler.fit(train_X)
      #train_X = scaler.transform(train_X)
      #test_X = scaler.transform(test_X)

      lstm_model= build_LSTM(sequence_length, num_features)
  # check point the model with lowest validation loss
  #checkpoint= keras.callbacks.ModelCheckpoint("final_project_lstm.keras",save_best_only=True)
      # DCB: fix # ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/content/lstm_cv_fold1

      checkpoint= keras.callbacks.ModelCheckpoint( filepath="/content/lstm_cv_fold"+str(fold_no)+".keras",save_best_only=True)

  #stop trianing if validation loss does not imrove for 5 consecutive epochs
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-3, restore_best_weights=True)

      lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9)

      opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
      lstm_model.compile(loss="binary_crossentropy", metrics=['acc',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()], optimizer=opt)
  #opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
  #lstm_model.compile(optimizer=opt, loss="mse", metrics=["mae"])
      history = lstm_model.fit(train_tf,
      epochs=50,
      validation_data=val_tf,
      callbacks=[checkpoint, early_stopping])

      #lstm_hist = lstm_hist.append(history.history,ignore_index=True)
      lstm_hist = pd.concat([lstm_hist,pd.DataFrame.from_dict(history.history)], axis=0, ignore_index=True)
      #lstm_hist = pd.concat(history.history, ignore_index=True)
      lstm_best_score.append(checkpoint.best)

      loss = history.history['loss']
      val_loss = history.history['val_loss']
      epochs = range(1, len(loss) + 1)
      plt.figure()
      plt.plot(epochs, loss, 'bo', label='Training loss ESM2 Fold-'+str(fold_no))
      plt.plot(epochs, val_loss, 'b', label='Validation loss ESM2 Fold-'+str(fold_no))
      plt.title('Training and validation loss ESM2 Fold-'+str(fold_no))
      plt.legend()
      plt.show()

      #scores = model2.evaluate(test_X, Y[test], verbose=0)
      #acc_per_fold.append(scores[1] * 100)

      val_metrics = lstm_model.evaluate(val_tf, verbose=0)
      val_metrics_fold.append(val_metrics)

      test_metrics = lstm_model.evaluate(test_tf, verbose=0)
      test_metrics_fold.append(test_metrics)

      fold_no = fold_no + 1

  validation_cv_metrics_best = []
  test_cv_metrics_best = []
  val_f1_cv_best = []
  test_f1_cv_best = []
  for i in range(10):

    # DCB: fix # ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/content/lstm_cv_fold1
    lstm_model_CV = keras.models.load_model("/content/lstm_cv_fold"+str(i+1)+".keras")
    validation_cv_metrics= lstm_model_CV.evaluate(val_tf, verbose=0)
    validation_cv_metrics_best.append(validation_cv_metrics)
    val_f1 = f1_score(validation_cv_metrics[2], validation_cv_metrics[3])
    val_f1_cv_best.append(val_f1)
    test_cv_metrics= lstm_model_CV.evaluate(test_tf, verbose=0)
    test_cv_metrics_best.append(test_cv_metrics)
    test_f1 = f1_score(test_cv_metrics[2], test_cv_metrics[3])
    test_f1_cv_best.append(test_f1)

  for i in range(len(validation_cv_metrics_best)):
    print("Validation Precision Fold"+str(i+1)+": ", validation_cv_metrics_best[i][2])
    print("Validation Recall Fold"+str(i+1)+": ", validation_cv_metrics_best[i][3])
    print("Validation F1 Fold"+str(i+1)+": ", val_f1_cv_best[i])

  for i in range(len(test_cv_metrics_best)):
    print("Test Precision Fold"+str(i+1)+": ", test_cv_metrics_best[i][2])
    print("Test Recall Fold"+str(i+1)+": ", test_cv_metrics_best[i][3])
    print("Test F1 Fold"+str(i+1)+": ", test_f1_cv_best[i])

  val_metrics_best_mean = np.mean(np.array(validation_cv_metrics_best),axis=0)
  print(val_metrics_best_mean)
  print("Validation mean CV Precision: ", val_metrics_best_mean[2] )
  print("Validation mean CV Recall: ", val_metrics_best_mean[3] )
  val_F1_best_mean = np.mean(np.array(val_f1_cv_best),axis=0)
  print("Validation mean CV F1-Score: ", val_F1_best_mean)

  test_metrics_best_mean = np.mean(np.array(test_cv_metrics_best),axis=0)
  print(test_metrics_best_mean)
  print("Test mean CV Precision: ", test_metrics_best_mean[2] )
  print("Test mean CV Recall: ", test_metrics_best_mean[3] )
  test_F1_best_mean = np.mean(np.array(test_f1_cv_best),axis=0)
  print("Test mean CV F1-Score: ", test_F1_best_mean)

def LSTM_No_CV(x,y, sequence_length, num_features):

  X_train_array, X_test_array, y_train_array, y_test_array = split_XY_arrays(x, y)
  print("X_train Shape: ", X_train_array.shape)
  print("X_test Shape: ", X_test_array.shape)
  test_tf, test_tf_length = create_tf_dataset(X_test_array, y_test_array, 200)
  print("test length: ", test_tf_length)

  X_training_array, X_val_array, y_training_array, y_val_array = split_XY_arrays(X_train_array, y_train_array)
  print("X_training Shape: ", X_training_array.shape)
  print("X_val Shape: ", X_val_array.shape)

  lstm_model= build_LSTM(sequence_length, num_features)
  # check point the model with lowest validation loss
  #checkpoint= keras.callbacks.ModelCheckpoint("final_project_lstm.keras",save_best_only=True)
  # DCB: fix # ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/content/lstm_cv_fold1
  checkpoint= keras.callbacks.ModelCheckpoint( filepath="/content/lstm_no_cv.keras",save_best_only=True)

  #stop trianing if validation loss does not imrove for 5 consecutive epochs
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-3, restore_best_weights=True)

  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9)

  opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  lstm_model.compile(loss="binary_crossentropy", metrics=['acc',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()], optimizer=opt)
  #opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
  #lstm_model.compile(optimizer=opt, loss="mse", metrics=["mae"])
  history = lstm_model.fit(X_training_array, y_training_array,
  epochs=50,
  validation_data=(X_val_array,y_val_array),
  callbacks=[checkpoint, early_stopping])

  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()

  #Validation metrics

  # DCB: fix # ValueError: The filepath provided must end in `.keras` (Keras model format). Received: filepath=/content/lstm_cv_fold1
  lstm_model = keras.models.load_model("/content/lstm_no_cv.keras")

  # evaluate the model
  #loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)
  # evaluate the model
  val_loss, val_accuracy, val_precision, val_recall = lstm_model.evaluate(X_val_array,y_val_array, verbose=0)
  #print(val_loss, val_accuracy, val_precision, val_recall)
  print("Validation precision:",val_precision )
  print("Validation recall: ", val_recall)

  val_f1 = f1_score(val_precision, val_recall)
  print("Validation F1: ", val_f1)

  #Test metrcis

  test_loss, test_accuracy, test_precision, test_recall = lstm_model.evaluate(X_test_array,y_test_array, verbose=0)
  #print(test_loss, test_accuracy, test_precision, test_recall)
  print("Test precision:",test_precision )
  print("Testrecall: ", test_recall)

  test_f1 = f1_score(test_precision, test_recall)
  print("Test F1: ", test_f1)


  

def shap_plots_testing(x,y, df_shuffled_test):

  X_train_array, X_test_array, y_train_array, y_test_array = split_XY_arrays(x, y)
  print("X_train Shape: ", X_train_array.shape)
  print("X_test Shape: ", X_test_array.shape)
  test_tf, test_tf_length = create_tf_dataset(X_test_array, y_test_array, 200)
  print("test length: ", test_tf_length)

  X_training_array, X_val_array, y_training_array, y_val_array = split_XY_arrays(X_train_array, y_train_array)
  print("X_training Shape: ", X_training_array.shape)
  print("X_val Shape: ", X_val_array.shape)

  import tensorflow as tf
  tf.compat.v1.disable_v2_behavior()
  from tensorflow import keras
  import numpy as np

  
  # DCB: ValueError (need .keras)
  # lstm_model = keras.models.load_model("/content/lstm_no_cv")
  lstm_model = keras.models.load_model("/content/lstm_no_cv.keras")

  #print("test shape", X_test_650MB_array.shape)
  #print("training array shape", X_training_650MB_array.shape )

  import shap

  #explainer = shap.DeepExplainer(lstm_model_650MB, X_training_650MB_array[:100])
  explainer = shap.DeepExplainer(lstm_model, X_training_array)

  shap_values1 = explainer.shap_values(X_test_array, check_additivity=False)

  import matplotlib
  shap.initjs()
  for i in range(len(df_shuffled_test)):
    CDRH3_String= df_shuffled_test.iloc[i,4]
    binding = df_shuffled_test.iloc[i,0]
    #number_of_none_required = max_CDRH3_length - len(CDRH3_String)
    #print("Index : " +str(i) + " CDRH3 - ", CDRH3_String)
    print(str(i+1) + " CDRH3: ", CDRH3_String + ", Actual Binding: ", binding)
    cdrh3_aa_list = []
    m=1
    for aa in CDRH3_String:
      cdrh3_aa_list.append(aa + str(m))
      m=m+1
    #print(cdrh3_aa_list)
    #for k in range(number_of_none_required):
    #  cdrh3_aa_list.append('NONE' + str(m))
    #  m=m+1
    #print(cdrh3_aa_list)
    #array_shap_values1_mean=np.mean(np.array(shap_values1[0][i]), axis=1)
    #array_shap_values1 = np.array(shap_values1[0][i])
    #Ifthi: I am modifiying this as we shap_values is 4 dimensional array now
    array_shap_values1 = np.array(shap_values1[i])
    array_shap_values1_sum=np.sum(array_shap_values1[0:len(CDRH3_String)], axis=1)
    array_shap_values1_sum_reduced = array_shap_values1_sum[:,0]
    #Use display if you are indenting the shap.force_plot
    #display(shap.force_plot(explainer.expected_value[0], array_shap_values1_mean,cdrh3_aa_list))
    #Ifthi: we don't need display as we use matplotlib otherwise 'None' will be printed after each plot
    #(shap.force_plot(explainer.expected_value[0], array_shap_values1_sum,cdrh3_aa_list,matplotlib=matplotlib))
    (shap.force_plot(explainer.expected_value[0], array_shap_values1_sum_reduced,cdrh3_aa_list,matplotlib=matplotlib))
